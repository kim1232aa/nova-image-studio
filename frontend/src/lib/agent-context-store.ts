// Agent 模式自建上下文系统的 IndexedDB 持久化层
// 默认会话数据库: nova-agent-db (v1)；其他会话: nova-agent-db-${id} (v1)
//   store: messages (keyPath 'id')        —— 对话消息，靠 createdAt 排序
//   store: images   (keyPath 'imgId')     —— 图片登记表（仅描述 + 缩略图 + 字节引用）
//   store: meta      (keyPath 'key')       —— 会话元信息（模型选择等）
// 图片真实字节不在这里，存于 nova-image-db 的 blobs store（复用 image-downloader）。

import { storeImageBlob, getStoredBlob, deleteStoredBlobs } from '@/lib/image-downloader';
import { normalizeProductKey } from '@/lib/agent-chat-config';
import type { AgentMessage, AgentImageRecord, AgentProposal } from '@/lib/agent-chat-config';
import type { GptImageBackground, GptImageQuality, GptImageStyle } from '@/lib/model-capabilities';

const DB_NAME = 'nova-agent-db';
const DEFAULT_SESSION_ID = 'default';
const DB_VERSION = 1;
const MESSAGES_STORE = 'messages';
const IMAGES_STORE = 'images';
const META_STORE = 'meta';

let currentSessionId = DEFAULT_SESSION_ID;
const dbCache = new Map<string, IDBDatabase>();
const dbOpenPromises = new Map<string, Promise<IDBDatabase | null>>();

function getSessionDbName(id: string): string {
  return id === DEFAULT_SESSION_ID ? DB_NAME : `${DB_NAME}-${id}`;
}

function resolveSessionId(sessionId?: string): string {
  return sessionId ?? currentSessionId;
}

/** 非默认会话的 blob 使用独立命名空间；默认会话保留旧 key 以兼容已有数据。 */
function getAgentBlobJobId(imgId: string, sessionId: string): string {
  return sessionId === DEFAULT_SESSION_ID
    ? imgId
    : `agent-session-${encodeURIComponent(sessionId)}-${imgId}`;
}

/** 选择后续 Agent 上下文读写所使用的会话数据库。 */
export function setAgentSession(id: string): void {
  currentSessionId = id;
}

/** 删除非默认会话数据库及其图片字节；任何失败都向调用方暴露。 */
export async function deleteAgentSessionDatabase(id: string): Promise<void> {
  if (id === DEFAULT_SESSION_ID) {
    throw new Error('默认会话数据库不可删除');
  }
  if (typeof indexedDB === 'undefined') {
    throw new Error('IndexedDB 不可用，无法删除会话数据库');
  }

  const dbName = getSessionDbName(id);
  let db = dbCache.get(dbName);
  if (!db) {
    const pendingOpen = dbOpenPromises.get(dbName);
    if (pendingOpen) db = await pendingOpen ?? undefined;
  }
  if (!db) db = await openAgentDB(id) ?? undefined;
  if (!db) throw new Error(`无法打开会话数据库: ${id}`);

  const images = await getAllStrict<AgentImageRecord>(db, IMAGES_STORE);
  if (dbCache.get(dbName) === db) dbCache.delete(dbName);
  dbOpenPromises.delete(dbName);
  try { db.close(); } catch { /* ignore */ }

  await new Promise<void>((resolve, reject) => {
    const req = indexedDB.deleteDatabase(dbName);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error || new Error(`删除会话数据库失败: ${id}`));
    req.onblocked = () => reject(new Error(`删除会话数据库被阻塞: ${id}`));
  });

  // 只有数据库删除成功后才清理共享 blob，失败时保留会话数据可重试。
  await Promise.all(images.map(image => deleteAgentImageBytes(image.imgId, id)));
}

function openAgentDB(sessionId: string): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);

  const dbName = getSessionDbName(sessionId);
  const cachedDb = dbCache.get(dbName);
  if (cachedDb) return Promise.resolve(cachedDb);

  const pendingOpen = dbOpenPromises.get(dbName);
  if (pendingOpen) return pendingOpen;

  const dbPromise = new Promise<IDBDatabase | null>((resolve) => {
    const req = indexedDB.open(dbName, DB_VERSION);
    req.onerror = () => {
      dbOpenPromises.delete(dbName);
      resolve(null);
    };
    req.onsuccess = () => {
      const db = req.result;
      const invalidate = () => {
        if (dbCache.get(dbName) === db) dbCache.delete(dbName);
      };
      db.onversionchange = () => {
        try { db.close(); } catch { /* ignore */ }
        invalidate();
      };
      db.onclose = invalidate;
      dbCache.set(dbName, db);
      dbOpenPromises.delete(dbName);
      resolve(db);
    };
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(MESSAGES_STORE)) {
        db.createObjectStore(MESSAGES_STORE, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(IMAGES_STORE)) {
        db.createObjectStore(IMAGES_STORE, { keyPath: 'imgId' });
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'key' });
      }
    };
  });
  dbOpenPromises.set(dbName, dbPromise);
  return dbPromise;
}

function getAll<T>(db: IDBDatabase, storeName: string): Promise<T[]> {
  return new Promise((resolve) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve((req.result as T[]) || []);
    req.onerror = () => resolve([]);
  });
}

function getAllStrict<T>(db: IDBDatabase, storeName: string): Promise<T[]> {
  return new Promise((resolve, reject) => {
    try {
      const tx = db.transaction(storeName, 'readonly');
      const req = tx.objectStore(storeName).getAll();
      req.onsuccess = () => resolve((req.result as T[]) || []);
      req.onerror = () => reject(req.error || new Error(`读取 ${storeName} 失败`));
      tx.onerror = () => reject(tx.error || new Error(`读取 ${storeName} 事务失败`));
    } catch (error) {
      reject(error);
    }
  });
}

// ===== 加载完整会话 =====

/**
 * 旧版 CDP 抓图只把商品标题/来源写进消息，图片记录本身缺少 productKey/productName。
 * 从持久化消息恢复作用域，让旧会话也能按商品分组并按模型上限自动选图。
 */
export function backfillProductScopes(
  messages: AgentMessage[],
  images: AgentImageRecord[],
): { images: AgentImageRecord[]; changedIds: string[] } {
  const scopeByImageId = new Map<string, { productKey: string; productName: string }>();
  const pattern = /已从浏览器抓取商品《([^》]+)》\s*\d+\s*张图并登记：([^（\n]+)（来源：(https?:\/\/[^）\s]+)）/g;

  for (const message of messages) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(message.text || '')) !== null) {
      const productName = match[1].trim();
      const productKey = normalizeProductKey(match[3]) || match[3].trim();
      const ids = match[2].split(/[、,，\s]+/).map(id => id.trim()).filter(Boolean);
      for (const imgId of ids) scopeByImageId.set(imgId, { productKey, productName });
    }
  }

  const changedIds: string[] = [];
  const migrated = images.map(image => {
    if (image.productKey) return image;
    const scope = scopeByImageId.get(image.imgId);
    if (!scope) return image;
    changedIds.push(image.imgId);
    return { ...image, ...scope };
  });
  return { images: migrated, changedIds };
}

export interface AgentSessionSnapshot {
  messages: AgentMessage[];
  images: AgentImageRecord[];
  imageModel: string | null;
}

export async function loadAgentSession(sessionId?: string): Promise<AgentSessionSnapshot> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return { messages: [], images: [], imageModel: null };

  const [messages, images, meta] = await Promise.all([
    getAll<AgentMessage>(db, MESSAGES_STORE),
    getAll<AgentImageRecord>(db, IMAGES_STORE),
    getAll<{ key: string; value: string }>(db, META_STORE),
  ]);

  messages.sort((a, b) => a.createdAt - b.createdAt);
  images.sort((a, b) => a.createdAt - b.createdAt);
  const migrated = backfillProductScopes(messages, images);
  if (migrated.changedIds.length > 0) {
    const changed = new Set(migrated.changedIds);
    const tx = db.transaction(IMAGES_STORE, 'readwrite');
    const store = tx.objectStore(IMAGES_STORE);
    for (const image of migrated.images) {
      if (changed.has(image.imgId)) store.put(image);
    }
  }
  const imageModel = meta.find(item => item.key === 'imageModel')?.value ?? null;

  return { messages, images: migrated.images, imageModel };
}

// ===== 消息读写 =====

export async function putMessage(message: AgentMessage, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(MESSAGES_STORE, 'readwrite');
    tx.objectStore(MESSAGES_STORE).put(message);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== 图片登记表读写 =====

export async function putImageRecord(record: AgentImageRecord, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(IMAGES_STORE, 'readwrite');
    tx.objectStore(IMAGES_STORE).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== 元信息 =====

export async function saveImageModel(model: string, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).put({ key: 'imageModel', value: model });
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== 撤回消息 =====

export async function deleteMessages(ids: string[], sessionId?: string): Promise<void> {
  if (ids.length === 0) return;
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(MESSAGES_STORE, 'readwrite');
    const store = tx.objectStore(MESSAGES_STORE);
    for (const id of ids) store.delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/** 从当前 Agent 会话数据库中删除图片登记记录 */
export async function deleteImageRecords(imgIds: string[], sessionId?: string): Promise<void> {
  if (imgIds.length === 0) return;
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(IMAGES_STORE, 'readwrite');
    const store = tx.objectStore(IMAGES_STORE);
    for (const id of imgIds) store.delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/** 从 nova-image-db 中删除 agent 图片的 blob 字节 */
export async function deleteAgentImageBytes(imgId: string, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  await deleteStoredBlobs(getAgentBlobJobId(imgId, session), 1);
}

// ===== 清空会话（清空重开） =====

export async function clearAgentSession(sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  const images = await getAll<AgentImageRecord>(db, IMAGES_STORE);
  await Promise.all(images.map(image => deleteAgentImageBytes(image.imgId, session)));

  return new Promise((resolve) => {
    const tx = db.transaction([MESSAGES_STORE, IMAGES_STORE, META_STORE], 'readwrite');
    tx.objectStore(MESSAGES_STORE).clear();
    tx.objectStore(IMAGES_STORE).clear();
    tx.objectStore(META_STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== Pending Proposal 持久化（刷新恢复「等待你确认」状态）=====
// 将待确认的提案、分析文本、推理文本和 reedit 标志存入 meta store，
// 页面刷新后自动恢复 proposal 阶段，避免丢失。

export interface PendingProposalData {
  proposal: AgentProposal;
  pendingAnalysis: string;
  pendingReasoning: string;
  isReedit: boolean;
  /** 当前提案之后排队等待确认的商品提案；刷新后随当前提案一起恢复 */
  queuedProposals?: AgentProposal[];
}

const PENDING_PROPOSAL_KEY = 'pendingProposal';

export async function savePendingProposal(data: PendingProposalData, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).put({ key: PENDING_PROPOSAL_KEY, value: JSON.stringify(data) });
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

export async function loadPendingProposal(sessionId?: string): Promise<PendingProposalData | null> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return null;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readonly');
    const req = tx.objectStore(META_STORE).get(PENDING_PROPOSAL_KEY);
    req.onsuccess = () => {
      const entry = req.result as { key: string; value: string } | undefined;
      if (!entry?.value) { resolve(null); return; }
      try {
        resolve(JSON.parse(entry.value) as PendingProposalData);
      } catch {
        resolve(null);
      }
    };
    req.onerror = () => resolve(null);
  });
}

export async function clearPendingProposal(sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).delete(PENDING_PROPOSAL_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== Pending Generation 持久化（刷新恢复「正在生图」状态）=====
// 将 taskId、proposal、分析文本等存入 meta store，
// 页面刷新后自动恢复轮询，避免生成中的图片丢失。

export interface PendingGenerationData {
  taskId: string;
  proposal: AgentProposal;
  pendingAnalysis: string;
  pendingReasoning: string;
  selectedImageIds: string[];
  model: string;
  outputSize: string;
  customSize?: string;
  aspectRatio: string;
  temperature: number;
  gptImageQuality?: GptImageQuality;
  gptImageStyle?: GptImageStyle;
  gptImageBackground?: GptImageBackground;
  parallelCount: number;
  startedAt: number;
}

const PENDING_GENERATION_KEY = 'pendingGeneration';

export async function savePendingGeneration(data: PendingGenerationData, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).put({ key: PENDING_GENERATION_KEY, value: JSON.stringify(data) });
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

export async function loadPendingGeneration(sessionId?: string): Promise<PendingGenerationData | null> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return null;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readonly');
    const req = tx.objectStore(META_STORE).get(PENDING_GENERATION_KEY);
    req.onsuccess = () => {
      const entry = req.result as { key: string; value: string } | undefined;
      if (!entry?.value) { resolve(null); return; }
      try {
        resolve(JSON.parse(entry.value) as PendingGenerationData);
      } catch {
        resolve(null);
      }
    };
    req.onerror = () => resolve(null);
  });
}

export async function clearPendingGeneration(sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return;

  return new Promise((resolve) => {
    const tx = db.transaction(META_STORE, 'readwrite');
    tx.objectStore(META_STORE).delete(PENDING_GENERATION_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

// ===== 图片字节存取（复用 nova-image-db 的 blobs store）=====
// 默认会话使用历史 imgId key；其他会话用 sessionId 命名空间隔离。

export async function storeAgentImageBytes(imgId: string, blob: Blob, sessionId?: string): Promise<void> {
  const session = resolveSessionId(sessionId);
  await storeImageBlob(getAgentBlobJobId(imgId, session), 0, blob);
}

/** 查询 nova-upload-cache 中缓存的图片记录 */
interface UploadCacheRecord {
  key: string;
  name: string;
  mimeType: string;
  dataUrl: string;
  originalSize: number;
  processedSize: number;
  width: number;
  height: number;
  createdAt: number;
}

function openUploadCacheDB(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = indexedDB.open('nova-upload-cache', 1);
    req.onerror = () => resolve(null);
    req.onsuccess = () => resolve(req.result);
  });
}

function getFromUploadCache(db: IDBDatabase, key: string): Promise<UploadCacheRecord | null> {
  return new Promise((resolve) => {
    const tx = db.transaction('images', 'readonly');
    const req = tx.objectStore('images').get(key);
    req.onsuccess = () => resolve((req.result as UploadCacheRecord) || null);
    req.onerror = () => resolve(null);
  });
}

/** 从当前会话数据库的 images store 中查询单条图片登记记录 */
export async function getAgentImageRecord(imgId: string, sessionId?: string): Promise<AgentImageRecord | null> {
  const session = resolveSessionId(sessionId);
  const db = await openAgentDB(session);
  if (!db) return null;
  return new Promise((resolve) => {
    const tx = db.transaction(IMAGES_STORE, 'readonly');
    const req = tx.objectStore(IMAGES_STORE).get(imgId);
    req.onsuccess = () => resolve((req.result as AgentImageRecord) || null);
    req.onerror = () => resolve(null);
  });
}

export async function getAgentImageBytes(imgId: string, sessionId?: string): Promise<Blob | null> {
  const session = resolveSessionId(sessionId);
  // 1) 先查 nova-upload-cache（上传图片已压缩缓存于此，与其余模式共享）
  const record = await getAgentImageRecord(imgId, session);
  if (record?.contentHash) {
    try {
      const cacheDb = await openUploadCacheDB();
      if (cacheDb) {
        const cached = await getFromUploadCache(cacheDb, record.contentHash);
        cacheDb.close();
        if (cached?.dataUrl) {
          const base64 = cached.dataUrl.includes(',') ? cached.dataUrl.split(',')[1] : cached.dataUrl;
          if (base64) {
            const mime = cached.mimeType || 'image/png';
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return new Blob([bytes], { type: mime });
          }
        }
      }
    } catch {
      // 读取上传缓存失败时静默降级到 nova-image-db
    }
  }
  // 2) 降级到 nova-image-db（生成图片走此路径）。默认 key 保持旧数据兼容。
  if (session === DEFAULT_SESSION_ID) return getStoredBlob(imgId, 0);
  return getStoredBlob(getAgentBlobJobId(imgId, session), 0);
}

/** 把图片字节转成可直接喂给生图后端的 base64（不含 data: 前缀）
 *
 * 延迟下载支持：如果图片记录包含 remoteUrl 但本地无字节，则按需下载后返回 base64
 */
export async function getAgentImageBase64(imgId: string, sessionId?: string): Promise<{ data: string; mimeType: string } | null> {
  const session = resolveSessionId(sessionId);
  // 1) 先尝试从本地读取
  const blob = await getAgentImageBytes(imgId, session);
  if (blob) {
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
    const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
    return { data: base64, mimeType: blob.type || 'image/png' };
  }

  // 2) 本地无字节，检查是否有 remoteUrl（CDP 抓图等延迟下载场景）
  const sessionData = await loadAgentSession(session);
  const record = sessionData.images.find(r => r.imgId === imgId);
  if (record?.remoteUrl) {
    try {
      // 按需下载远程图片
      const response = await fetch(record.remoteUrl);
      if (!response.ok) return null;
      const downloadedBlob = await response.blob();

      // 下载后编码为 base64
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ''));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(downloadedBlob);
      });
      const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;

      // 可选：下载后缓存到本地（省得下次再下载）
      await storeAgentImageBytes(imgId, downloadedBlob, session);

      return { data: base64, mimeType: downloadedBlob.type || record.mimeType || 'image/jpeg' };
    } catch {
      return null; // 下载失败静默返回 null
    }
  }

  return null;
}
