'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { hasAnyApiKey } from '@/lib/settings-storage';
import { generateUUID } from '@/lib/uuid';
import { createNovaTask, getNovaTask, resolveImageTaskProvider, type ImageReference } from '@/lib/ccode-task-client';
import { fetchImageAsBlob } from '@/lib/image-downloader';
import {
  getGptImageAdvancedParamsForModel,
  resolveAgentModel,
  resolveSubmitLayout,
  type AgentModelCatalogEntry,
  type AgentResolvedLayout,
} from '@/lib/model-capabilities';
import type { ModelId } from '@/lib/gemini-config';
import {
  getCompleteImageModels,
  getDefaultImageModel,
  getImageModelById,
  loadRegistry,
} from '@/lib/nova-models';
import {
  streamAgentChat,
  describeImage,
  type StreamAgentHandle,
} from '@/lib/agent-chat-client';
import { executeAgentCdpTool } from '@/lib/agent-cdp-tools';
import {
  AGENT_DEFAULT_IMAGE_MODEL_FALLBACK,
  extractProductLinks,
  normalizeProductKey,
  type AgentMessage,
  type AgentImageRecord,
  type AgentProposal,
} from '@/lib/agent-chat-config';
import {
  loadAgentSession,
  putMessage,
  putImageRecord,
  saveImageModel,
  clearAgentSession,
  storeAgentImageBytes,
  getAgentImageBase64,
  deleteMessages,
  deleteImageRecords,
  deleteAgentImageBytes,
  savePendingProposal,
  loadPendingProposal,
  clearPendingProposal,
  savePendingGeneration,
  loadPendingGeneration,
  clearPendingGeneration,
  type PendingGenerationData,
} from '@/lib/agent-context-store';
import { getDefaultConfiguredTextModel } from '@/lib/model-endpoints';
import { supportsAgentNativeWebSearch } from '@/lib/nova-text-protocol';

export type AgentPhase = 'idle' | 'loading' | 'describing' | 'streaming' | 'proposal' | 'generating';

export type AgentCheckResult = 'idle' | 'completed' | 'processing' | 'queued' | 'failed' | 'error';

export interface AgentGenerationDraft {
  analysis: string;
  reasoning?: string;
  prompt: string;
  parallelCount: number;
  taskId?: string;
  startedAt: number;
}

export interface PendingUpload {
  id: string;
  name: string;
  preview: string;
  dataUrl: string;
  mimeType: string;
  badge?: string;
  source?: AgentImageRecord['source'];
}

const PREVIEW_MAX_SIDE = 512;

function isStoppedError(error: unknown): boolean {
  return error instanceof Error && error.message === '已停止';
}

/** 构建当前可用的图像模型目录，供 Agent 选择模型 */
function buildModelCatalog(): AgentModelCatalogEntry[] {
  return getCompleteImageModels(loadRegistry()).map(m => ({
    id: m.id,
    name: m.name,
    maxOutputSize: m.maxOutputSize,
  }));
}

/**
 * Agent 生图模型必须落在用户注册表里，否则确认生图时会报「未找到图片模型配置」。
 * 硬编码兜底（gemini-3-pro-image-preview）不在注册表时，改用设置里的默认图像模型。
 */
function resolveValidAgentImageModel(preferred?: string | null): ModelId {
  const registry = loadRegistry();
  if (preferred && getImageModelById(registry, preferred)) return preferred as ModelId;
  const configured =
    getDefaultImageModel(registry, 'imageToImage') ||
    getDefaultImageModel(registry, 'textToImage') ||
    getCompleteImageModels(registry)[0];
  return (configured?.id || AGENT_DEFAULT_IMAGE_MODEL_FALLBACK) as ModelId;
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mimeType });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('图片加载失败'));
    img.src = src;
  });
}


/** 从 Blob 直接生成缩略图 dataUrl，避免全尺寸 base64 转换 */
async function makePreviewFromBlob(blob: Blob): Promise<{ dataUrl: string; width: number; height: number }> {
  try {
    const blobUrl = URL.createObjectURL(blob);
    const img = await loadImage(blobUrl);
    URL.revokeObjectURL(blobUrl);
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (w <= PREVIEW_MAX_SIDE && h <= PREVIEW_MAX_SIDE) {
      // 小图直接转 dataUrl（尺寸小，不影响性能）
      const smallDataUrl = await blobToDataUrl(blob);
      return { dataUrl: smallDataUrl, width: w, height: h };
    }
    const scale = PREVIEW_MAX_SIDE / Math.max(w, h);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      const fallback = await blobToDataUrl(blob);
      return { dataUrl: fallback, width: w, height: h };
    }
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return { dataUrl: canvas.toDataURL('image/jpeg', 0.85), width: w, height: h };
  } catch {
    const fallback = await blobToDataUrl(blob);
    return { dataUrl: fallback, width: 0, height: 0 };
  }
}

function parseImgSeq(imgId: string): number {
  const match = imgId.match(/^img_(\d+)$/);
  return match ? Number(match[1]) : 0;
}

/**
 * 按最后一个上下文分隔点切片：分隔点之前的对话和图片对模型不可见。
 * 界面仍展示全部消息，这里只影响喂给模型的上下文。
 */
function sliceActiveContext(
  history: AgentMessage[],
  catalog: AgentImageRecord[],
): { history: AgentMessage[]; catalog: AgentImageRecord[] } {
  let dividerIndex = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === 'context-divider') { dividerIndex = i; break; }
  }
  if (dividerIndex === -1) return { history, catalog };

  const dividerAt = history[dividerIndex].createdAt;
  return {
    history: history.slice(dividerIndex + 1),
    catalog: catalog.filter(img => img.createdAt > dividerAt),
  };
}

async function resultImageToBlob(ref: string): Promise<Blob> {
  if (ref.startsWith('URL:')) return fetchImageAsBlob(ref.slice(4));
  if (ref.startsWith('MULTI_URL:')) return fetchImageAsBlob(ref.slice(10).split('|||')[0]);
  if (ref.startsWith('data:')) {
    const base64 = ref.split(',')[1] || '';
    const mime = ref.slice(5).split(';')[0] || 'image/png';
    return base64ToBlob(base64, mime);
  }
  return base64ToBlob(ref, 'image/png');
}

export function useAgentChat(sessionId = 'default') {
  const sessionIdRef = useRef(sessionId);
  const [ready, setReady] = useState(false);
  const [hasApiKey] = useState(() => hasAnyApiKey());
  const [phase, setPhase] = useState<AgentPhase>('idle');
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [images, setImages] = useState<AgentImageRecord[]>([]);
  const [proposal, setProposal] = useState<AgentProposal | null>(null);
  /** 当前提案之后等待展示的独立商品提案；用户确认/取消当前项后自动推进。 */
  const [, setProposalQueue] = useState<AgentProposal[]>([]);
  const proposalQueueRef = useRef<AgentProposal[]>([]);
  /** 最近一条用户消息里的商品链接（归一化键）；非空表示当前处于「批量商品轮」，任务完成后允许自动续跑 */
  const userLinksRef = useRef<string[]>([]);
  /** 批量轮内已自动续跑的次数；上限只是防失控兜底，是否继续由模型对照原始需求自己判断 */
  const autoContinueCountRef = useRef(0);
  const messagesRef = useRef<AgentMessage[]>([]);
  const imagesRef = useRef<AgentImageRecord[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  useEffect(() => { imagesRef.current = images; }, [images]);
  const phaseRef = useRef<AgentPhase>(phase);
  useEffect(() => { phaseRef.current = phase; }, [phase]);
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [imageModel, setImageModelState] = useState<ModelId>(() => resolveValidAgentImageModel());
  const [error, setError] = useState<string | null>(null);
  const [generatingTaskId, setGeneratingTaskId] = useState<string | null>(null);
  const [generatingStartedAt, setGeneratingStartedAt] = useState<number | null>(null);
  const [generationDraft, setGenerationDraft] = useState<AgentGenerationDraft | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(() =>
    typeof localStorage !== 'undefined' ? localStorage.getItem('nova-agent-web-search') === 'true' : false
  );
  const [cdpEnabled, setCdpEnabled] = useState(() =>
    typeof localStorage !== 'undefined' ? localStorage.getItem('nova-agent-cdp') === 'true' : false
  );
  const [intentRecognition, setIntentRecognition] = useState(() =>
    typeof localStorage !== 'undefined' ? localStorage.getItem('nova-agent-intent-recognition') !== 'false' : true
  );

  const streamHandleRef = useRef<StreamAgentHandle | null>(null);
  const mountedRef = useRef(true);
  const pollControllersRef = useRef(new Map<string, { controller: AbortController; wake: () => void }>()).current;
  const pendingGenerationTaskRef = useRef<string | null>(null);
  const generationToResumeRef = useRef<PendingGenerationData | null>(null);
  const generationEpochRef = useRef(0);
  const describeAbortRef = useRef<AbortController | null>(null);
  const seqRef = useRef(0);
  /** 当模型返回提案时，暂存分析文本，等生图完成后与结果合并为一条消息 */
  const pendingAnalysisRef = useRef('');
  const pendingReasoningRef = useRef('');
  /** 标记当前提案是否来自"重新编辑"（而非新消息触发），用于取消时决定是否允许撤回 */
  const isReeditRef = useRef(false);
  /** 保存当前提案引用，生图完成后若 state proposal 已被清除时仍可获取 reason 等字段 */
  const proposalRef = useRef<AgentProposal | null>(null);
  /** 镜像 imageModel state，供 runChat 回调中同步读取 */
  const imageModelRef = useRef(imageModel);
  useEffect(() => { imageModelRef.current = imageModel; }, [imageModel]);

  const getAgentTextModelConfig = useCallback(() => {
    const configured = getDefaultConfiguredTextModel('agent');
    if (!configured?.apiKey || !configured.baseUrl || !configured.modelId) {
      throw new Error('请先在设置中完成 Agent 默认文本模型配置');
    }
    return configured;
  }, []);

  const agentSupportsWebSearch = useCallback(() => {
    const configured = getDefaultConfiguredTextModel('agent');
    if (!configured?.apiKey || !configured.baseUrl || !configured.modelId) {
      return false;
    }
    return supportsAgentNativeWebSearch(configured.protocol);
  }, []);

  // ===== 流式更新批处理（rAF 节流） =====
  const streamingTextBufRef = useRef('');
  const streamingReasoningBufRef = useRef('');
  const rafIdRef = useRef<number | null>(null);

  /** 刷新流式文本到 state（每帧调用一次） */
  const flushStreamingBuffers = useCallback(() => {
    rafIdRef.current = null;
    const text = streamingTextBufRef.current;
    const reasoning = streamingReasoningBufRef.current;
    streamingTextBufRef.current = '';
    streamingReasoningBufRef.current = '';
    if (!mountedRef.current) return;
    if (text) setStreamingText(prev => prev + text);
    if (reasoning) setStreamingReasoning(prev => prev + reasoning);
  }, []);

  /** 将 token 追加到缓冲区，并调度下一帧刷新 */
  const appendStreamingToken = useCallback((type: 'text' | 'reasoning', token: string) => {
    if (!mountedRef.current) return;
    if (type === 'text') streamingTextBufRef.current += token;
    else streamingReasoningBufRef.current += token;
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(flushStreamingBuffers);
    }
  }, [flushStreamingBuffers]);

  /** 立即刷新并取消待处理的 rAF（在 onDone/onReset/清理时调用） */
  const flushAndCancelRaf = useCallback(() => {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    const text = streamingTextBufRef.current;
    const reasoning = streamingReasoningBufRef.current;
    streamingTextBufRef.current = '';
    streamingReasoningBufRef.current = '';
    if (!mountedRef.current) return;
    if (text) setStreamingText(prev => prev + text);
    if (reasoning) setStreamingReasoning(prev => prev + reasoning);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    (async () => {
      const [session, pending, generation] = await Promise.all([
        loadAgentSession(sessionIdRef.current),
        loadPendingProposal(sessionIdRef.current),
        loadPendingGeneration(sessionIdRef.current),
      ]);
      if (cancelled || !mountedRef.current) return;
      setMessages(session.messages);
      setImages(session.images);
      seqRef.current = session.images.reduce((max, img) => Math.max(max, parseImgSeq(img.imgId)), 0);
      const validImageModel = resolveValidAgentImageModel(session.imageModel);
      imageModelRef.current = validImageModel;
      setImageModelState(validImageModel);

      if (pending) {
        // 恢复待确认的提案，使用户刷新后仍可看到「等待你确认」卡片
        pendingAnalysisRef.current = pending.pendingAnalysis;
        pendingReasoningRef.current = pending.pendingReasoning;
        isReeditRef.current = pending.isReedit;
        setProposal(pending.proposal);
        proposalQueueRef.current = pending.queuedProposals || [];
        setProposalQueue(pending.queuedProposals || []);
        setPhase('proposal');
      }

      if (generation) {
        // 恢复正在生图的状态：还原 taskId 和 refs，继续轮询结果
        pendingGenerationTaskRef.current = generation.taskId;
        pendingAnalysisRef.current = generation.pendingAnalysis;
        pendingReasoningRef.current = generation.pendingReasoning;
        proposalRef.current = generation.proposal;
        setGeneratingTaskId(generation.taskId);
        setGeneratingStartedAt(generation.startedAt);
        setGenerationDraft({
          analysis: generation.pendingAnalysis || generation.proposal.reason || '根据你的描述，正在生成图片。',
          reasoning: generation.pendingReasoning || undefined,
          prompt: generation.proposal.prompt,
          parallelCount: generation.parallelCount,
          taskId: generation.taskId,
          startedAt: generation.startedAt,
        });
        setPhase('generating');
        generationToResumeRef.current = generation;
      }

      setReady(true);
    })();
    return () => { cancelled = true; };
  }, []);

  const appendMessage = useCallback((message: AgentMessage) => {
    if (!mountedRef.current) return;
    // ref 同步追加：maybeAutoContinue 等同一 tick 内读 messagesRef 的路径，
    // 不能等到 render 后的 useEffect 才看见这条消息；useEffect 之后的整体回写与此一致，不会重复。
    messagesRef.current = [...messagesRef.current, message];
    setMessages(prev => [...prev, message]);
    void putMessage(message, sessionIdRef.current);
  }, []);

  const registerImage = useCallback((record: AgentImageRecord) => {
    if (!mountedRef.current) return;
    setImages(prev => [...prev, record]);
    void putImageRecord(record, sessionIdRef.current);
  }, []);

  const nextImgId = useCallback(() => {
    seqRef.current += 1;
    return `img_${seqRef.current}`;
  }, []);

  // 给一张图片建立登记：存字节 + 生成预览 + 视觉描述
  const ingestImage = useCallback(async (
    source: AgentImageRecord['source'],
    blob: Blob,
    previewDataUrl: string,
    mimeType: string,
    sourceTaskId?: string,
    dims?: { width: number; height: number },
    contentHash?: string,
    describeSignal?: AbortSignal,
  ): Promise<AgentImageRecord> => {
    if (!mountedRef.current) throw new Error('已停止');
    const imgId = nextImgId();
    // 上传图片（有 contentHash）已在 prepareUploadImage 时存于 nova-upload-cache，
    // 不再重复存到 nova-image-db，节省空间；生成图片无 contentHash 则照常存储。
    if (source === 'generated' || !contentHash) {
      await storeAgentImageBytes(imgId, blob, sessionIdRef.current);
      if (!mountedRef.current) throw new Error('已停止');
    }

    let description = '';
    try {
      const configured = getAgentTextModelConfig();
      description = await describeImage(
        configured.apiKey,
        configured.modelId,
        configured.protocol,
        previewDataUrl,
        describeSignal,
        configured.baseUrl,
      );
      if (!mountedRef.current) throw new Error('已停止');
    } catch (error) {
      if (!mountedRef.current) throw error;
      description = '(图片描述生成失败)';
    }

    const record: AgentImageRecord = {
      imgId,
      source,
      thumbnail: previewDataUrl,
      description: description || '(无描述)',
      mimeType,
      contentHash,
      sourceTaskId,
      width: dims?.width && dims.width > 0 ? dims.width : undefined,
      height: dims?.height && dims.height > 0 ? dims.height : undefined,
      createdAt: Date.now(),
    };
    registerImage(record);
    return record;
  }, [getAgentTextModelConfig, nextImgId, registerImage]);

  /**
   * CDP 工具执行器：执行浏览器工具；抓图工具返回的 localUrls 逐张登记进图片目录，
   * 并把新 imgId 列表拼回给模型，使其可以在 propose_image_action 里引用。
   *
   * 优化策略：
   * 1. 只存 URL + 缩略图，不立即下载完整图（省时间和空间）
   * 2. 将抓图结果作为 assistant 消息持久化，让模型记住已抓取的图片，避免重复抓取
   * 3. 生成图时才按需下载 URL 并编码 base64
   */
  const cdpExecutor = useCallback(async (
    name: string,
    args: Record<string, unknown>,
    onProgress?: (text: string) => void,
  ): Promise<string> => {
    if (!mountedRef.current) return '';
    const result = await executeAgentCdpTool(name, args, onProgress);
    if (!mountedRef.current) return result.text;
    if (!result.localUrls || result.localUrls.length === 0) return result.text;

    const sourceKey = typeof result.sourceKey === 'string' ? result.sourceKey.trim() : '';
    const sourceTitle = typeof result.sourceTitle === 'string' ? result.sourceTitle.trim() : '';
    const sourceUrl = typeof result.sourceUrl === 'string' ? result.sourceUrl.trim() : '';
    const ingestedIds: string[] = [];
    for (const localUrl of result.localUrls) {
      try {
        // 只下载缩略图，不下载完整图（延迟下载优化）
        const blob = await fetchImageAsBlob(localUrl);
        const preview = await makePreviewFromBlob(blob);

        // 只存缩略图 + URL，真实图片字节不存（按需下载）
        // 描述先填商品标题，保证模型能按商品分组选择参考图；视觉描述生成是独立增强，不在这里阻塞
        const record: AgentImageRecord = {
          imgId: nextImgId(),
          source: 'uploaded',
          thumbnail: preview.dataUrl,
          description: sourceTitle ? `商品《${sourceTitle}》的图` : '',
          mimeType: blob.type || 'image/jpeg',
          width: preview.width,
          height: preview.height,
          remoteUrl: localUrl, // 存 URL，生成时才下载
          productKey: normalizeProductKey(sourceKey || sourceUrl),
          productName: sourceTitle || undefined,
          createdAt: Date.now(),
        };
        registerImage(record);
        ingestedIds.push(record.imgId);
      } catch {
        // 单张图登记失败不阻塞其余图片
      }
    }
    let text = result.text;
    if (ingestedIds.length > 0) {
      text += `\n\n以上图片已登记进图片目录：${ingestedIds.join('、')}。你可以在 propose_image_action 的 referenced_image_ids 中引用它们。`;

      // 将抓图结果作为 assistant 消息持久化到会话历史
      // 这样模型下次对话时能看到"我已经抓过这些图"，避免重复抓取
      // 消息里带上商品标题和链接，模型才能分清哪些 img 属于哪个商品
      appendMessage({
        id: generateUUID(),
        role: 'assistant',
        text: `✓ 已从浏览器抓取${sourceTitle ? `商品《${sourceTitle}》` : ''} ${ingestedIds.length} 张图并登记：${ingestedIds.join('、')}${sourceUrl ? `（来源：${sourceUrl}）` : ''}`,
        createdAt: Date.now(),
      });
    }
    return text;
  }, [nextImgId, registerImage, appendMessage]);

  /** 重新生成已有图片的描述 */
  const redescribeImage = useCallback(async (imgId: string): Promise<string> => {
    const record = images.find(img => img.imgId === imgId);
    if (!record) throw new Error(`图片 ${imgId} 不存在`);
    const configured = getAgentTextModelConfig();
    const newDescription = await describeImage(
      configured.apiKey,
      configured.modelId,
      configured.protocol,
      record.thumbnail,
      undefined,
      configured.baseUrl,
    );
    if (!mountedRef.current) return newDescription || '(无描述)';
    const description = newDescription || '(无描述)';
    const updated: AgentImageRecord = { ...record, description };
    setImages(prev => prev.map(img => img.imgId === imgId ? updated : img));
    if (mountedRef.current) {
      void putImageRecord(updated, sessionIdRef.current);
    }
    return description;
  }, [getAgentTextModelConfig, images]);

  const persistStreamFailure = useCallback((message: string) => {
    if (!mountedRef.current) return;
    setError(message);
    appendMessage({
      id: generateUUID(),
      role: 'system-note',
      text: `请求失败：${message}`,
      createdAt: Date.now(),
    });
    setPhase('idle');
  }, [appendMessage]);

  const runChat = useCallback((history: AgentMessage[], catalog: AgentImageRecord[]) => {
    if (!mountedRef.current) return;
    let configured: ReturnType<typeof getAgentTextModelConfig>;
    try {
      configured = getAgentTextModelConfig();
    } catch (err) {
      persistStreamFailure(err instanceof Error ? err.message : '请求失败');
      return;
    }
    const modelCatalog = buildModelCatalog();
    setPhase('streaming');
    flushAndCancelRaf();
    setStreamingText('');
    setStreamingReasoning(cdpEnabled ? '正在准备浏览器工具…\n' : '');

    let reasoningBuf = cdpEnabled ? '正在准备浏览器工具…\n' : '';

    const handle = streamAgentChat(
      {
        apiKey: configured.apiKey,
        model: configured.modelId,
        protocol: configured.protocol,
        history,
        webSearch: webSearchEnabled && supportsAgentNativeWebSearch(configured.protocol),
        cdp: cdpEnabled,
        cdpExecutor: cdpEnabled ? cdpExecutor : undefined,
        catalog: catalog.map(img => ({ imgId: img.imgId, description: img.description })),
        modelCatalog,
      },
      {
        onDelta: token => appendStreamingToken('text', token),
        onReasoning: token => {
          reasoningBuf += token;
          appendStreamingToken('reasoning', token);
        },
        onToolActivity: text => {
          reasoningBuf += text;
          appendStreamingToken('reasoning', text);
        },
        onResetAttempt: () => {
          if (!mountedRef.current) return;
          reasoningBuf = '';
          flushAndCancelRaf();
          setStreamingText('');
          setStreamingReasoning('');
        },
        onDone: (fullText, parsedProposal, parsedProposals) => {
          if (!mountedRef.current) return;
          streamHandleRef.current = null;
          flushAndCancelRaf();
          setStreamingText('');
          setStreamingReasoning('');
          const text = fullText.trim();
          const reasoning = reasoningBuf.trim();
          if (parsedProposal) {
            // 模型自动选择：Agent 指定模型 id 或用户要求分辨率档位时自动切换
            const resolvedModel = resolveValidAgentImageModel(resolveAgentModel(
              imageModelRef.current,
              parsedProposal.requestedModelId,
              parsedProposal.requestedOutputSize,
              modelCatalog,
            ));
            if (resolvedModel !== imageModelRef.current) {
              imageModelRef.current = resolvedModel;
              setImageModelState(resolvedModel);
              if (mountedRef.current) {
                void saveImageModel(resolvedModel, sessionIdRef.current);
              }
            }
            // 有提案：不保存为单独消息，暂存分析文本供生图成功后合并
            pendingAnalysisRef.current = text;
            pendingReasoningRef.current = reasoning;
            isReeditRef.current = false;
            setProposal(parsedProposal);
            // 多商品：同轮返回的其余提案排队，当前项确认/取消后自动推进
            const rest = (parsedProposals || []).filter(item => item !== parsedProposal);
            proposalQueueRef.current = rest;
            setProposalQueue(rest);
            setPhase('proposal');
            // 持久化 pending proposal，刷新页面后可以恢复（含排队中的其余商品提案）
            void savePendingProposal({
              proposal: parsedProposal,
              pendingAnalysis: text,
              pendingReasoning: reasoning,
              isReedit: false,
              queuedProposals: rest,
            }, sessionIdRef.current);
          } else {
            // 纯文本回复必须落盘：模型只调工具不吐字时 text 为空，
            // 以前直接丢掉，界面就像「卡住后什么都没发生」。
            appendMessage({
              id: generateUUID(),
              role: 'assistant',
              text: text.length > 0
                ? text
                : (reasoning.length > 0
                  ? '浏览器操作已结束，但模型没有给出文字回复。请再发一条继续。'
                  : '模型没有返回内容。请重试一次。'),
              reasoning: reasoning.length > 0 ? reasoning : undefined,
              createdAt: Date.now(),
            });
            setPhase('idle');
          }
        },
        onError: err => {
          if (!mountedRef.current) return;
          streamHandleRef.current = null;
          flushAndCancelRaf();
          setStreamingText('');
          setStreamingReasoning('');
          persistStreamFailure(err.message || '请求失败');
        },
      },
      configured.baseUrl,
    );
    streamHandleRef.current = handle;
  }, [appendMessage, appendStreamingToken, flushAndCancelRaf, getAgentTextModelConfig, persistStreamFailure, webSearchEnabled, cdpEnabled, cdpExecutor]);

  const sendMessage = useCallback(async (text: string, uploads: PendingUpload[], imageReferences?: string[]) => {
    if (!mountedRef.current || !ready || phase !== 'idle') return;
    const trimmed = text.trim();
    if (trimmed.length === 0 && uploads.length === 0) return;
    setError(null);
    // 新的商品链接开启新一轮批量轮：重置自动续跑计数；不含链接的消息（如"继续"）不重置
    const links = extractProductLinks(trimmed);
    if (links.length > 0) {
      userLinksRef.current = links;
      autoContinueCountRef.current = 0;
    }
    // 用户发送新消息时，丢弃任何待定提案的分析文本
    pendingAnalysisRef.current = '';
    pendingReasoningRef.current = '';
    isReeditRef.current = false;
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }

    const uploadedRecords: AgentImageRecord[] = [];
    const linkedIds: string[] = [];
    if (uploads.length > 0) {
      const descController = new AbortController();
      describeAbortRef.current = descController;

      setPhase('describing');
      const seenHashes = new Set<string>();
      try {
        for (const upload of uploads) {
        if (!mountedRef.current) return;
        const hash = upload.id;
        // 同批内重复 + 历史已登记重复，统一按内容哈希复用，不重复登记
        if (hash && seenHashes.has(hash)) continue;
        const existing = hash
          ? [...images, ...uploadedRecords].find(img => img.contentHash === hash)
          : undefined;
        if (existing) {
          if (hash) seenHashes.add(hash);
          if (!linkedIds.includes(existing.imgId)) linkedIds.push(existing.imgId);
          continue;
        }
        try {
          const blob = await resultImageToBlob(upload.dataUrl);
          const preview = await makePreviewFromBlob(blob);
          const record = await ingestImage(upload.source || 'uploaded', blob, preview.dataUrl, upload.mimeType, undefined, { width: preview.width, height: preview.height }, hash || undefined, descController.signal);
          uploadedRecords.push(record);
          if (hash) seenHashes.add(hash);
          linkedIds.push(record.imgId);
        } catch (err) {
          if (mountedRef.current) {
            setError(err instanceof Error ? err.message : '图片处理失败');
          }
        }
        }
      } finally {
        if (describeAbortRef.current === descController) {
          describeAbortRef.current = null;
        }
      }
    }

    const uploadedIds = linkedIds;
    const refSuffix = imageReferences && imageReferences.length > 0
      ? `\n[引用图片: ${imageReferences.join(', ')}]`
      : '';
    const noteSuffix = uploadedIds.length > 0 ? `\n[已上传图片: ${uploadedIds.join(', ')}]` : '';
    // 多链接批量轮：显式提醒模型一次出齐所有商品提案（排队确认），避免只出一个再等续跑
    const batchSuffix = links.length > 1
      ? `\n[系统：本条含 ${links.length} 个商品链接。抓完所有商品后，在同一轮为每个商品各调用一次 propose_image_action（填各自的 product_key/product_name，参考图只用该商品的），一次出齐，不要只出一个。]`
      : '';
    const userMessage: AgentMessage = {
      id: generateUUID(),
      role: 'user',
      text: `${trimmed}${refSuffix}${noteSuffix}${batchSuffix}`.trim(),
      imageIds: uploadedIds.length > 0 ? uploadedIds : undefined,
      createdAt: Date.now(),
    };
    appendMessage(userMessage);

    const fullHistory = [...messages, userMessage];
    const fullCatalog = [...images, ...uploadedRecords];
    const { history, catalog } = sliceActiveContext(fullHistory, fullCatalog);
    runChat(history, catalog);
  }, [images, ingestImage, messages, phase, ready, appendMessage, runChat]);

  /** 批量轮里自动续跑的次数上限——纯粹防失控兜底，不是业务规则 */
  const AUTO_CONTINUE_LIMIT = 10;

  /**
   * 一项生成完成后自动续跑一轮。这里不做「已覆盖/剩余商品」之类的硬规则：
   * 模型看得见完整历史（原始要求、已抓的图、已出的方案、已完成的图），
   * 还有没有没做完的交给它自己判断——做完它自然会用文字回复，续跑随之停止。
   * 仅在用户消息带商品链接的批量轮里触发；普通单图请求不续跑。
   */
  const maybeAutoContinue = useCallback((): boolean => {
    if (userLinksRef.current.length === 0) return false;
    if (autoContinueCountRef.current >= AUTO_CONTINUE_LIMIT) return false;
    autoContinueCountRef.current += 1;
    const userMessage: AgentMessage = {
      id: generateUUID(),
      role: 'user',
      text: '上一张图已完成。对照我最开始的要求（有哪些链接、每个商品要几张、一共要几张）：如果还有没做完的，继续调用 propose_image_action 出下一个方案；如果已经全部做完，直接用文字告诉我完成了，不要再调用工具。',
      createdAt: Date.now(),
    };
    appendMessage(userMessage);
    // appendMessage 已同步 messagesRef，这里直接用，不再手动拼 userMessage（否则会重复）
    const { history, catalog } = sliceActiveContext(messagesRef.current, imagesRef.current);
    runChat(history, catalog);
    return true;
  }, [appendMessage, runChat]);

  const cancelProposal = useCallback(() => {
    setProposal(null);
    setPhase('idle');
    // 取消时如果有待定分析，保存为一条助手消息供用户回顾
    const analysis = pendingAnalysisRef.current;
    const analysisReasoning = pendingReasoningRef.current;
    pendingAnalysisRef.current = '';
    pendingReasoningRef.current = '';
    // 二次编辑取消时不标记为可撤回，防止误操作删除已有图片
    const wasReedit = isReeditRef.current;
    isReeditRef.current = false;
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }
    if (analysis) {
      appendMessage({
        id: generateUUID(),
        role: 'assistant',
        text: analysis,
        reasoning: analysisReasoning || undefined,
        createdAt: Date.now(),
      });
    }
    appendMessage({
      id: generateUUID(),
      role: 'system-note',
      text: wasReedit ? '已取消本次重新编辑。' : '已取消本次生图提案。',
      withdrawable: !wasReedit,
      createdAt: Date.now(),
    });
    // 队列中还有下一个商品提案时直接推进，用户不需要再说"继续"。
    // 注意：读写都在 setState updater 之外，避免 StrictMode 双调用导致消息/持久化重复。
    const [next, ...rest] = proposalQueueRef.current;
    if (next) {
      proposalQueueRef.current = rest;
      setProposalQueue(rest);
      setProposal(next);
      setPhase('proposal');
      if (mountedRef.current) {
        void savePendingProposal(
          { proposal: next, pendingAnalysis: '', pendingReasoning: '', isReedit: false, queuedProposals: rest },
          sessionIdRef.current,
        );
      }
      appendMessage({
        id: generateUUID(),
        role: 'system-note',
        text: `已切换到下一个商品的提案（剩余 ${rest.length} 个待确认）。`,
        createdAt: Date.now(),
      });
      return;
    }
    // 队列已空就到 idle 为止。取消是用户主动叫停，不再自动续跑——下一步交给用户自己说。
  }, [appendMessage]);

  // 撤回最后一轮对话：从最后一条用户消息起（含其后的助手回复与本提示）全部删除，避免污染后续上下文
  const withdrawTurn = useCallback((noteId: string) => {
    if (!mountedRef.current) return;
    setMessages(prev => {
      const noteIndex = prev.findIndex(m => m.id === noteId);
      if (noteIndex === -1) return prev;
      let start = noteIndex;
      for (let i = noteIndex - 1; i >= 0; i--) {
        if (prev[i].role === 'user') { start = i; break; }
      }
      const removed = prev.slice(start);
      if (mountedRef.current) {
        void deleteMessages(removed.map(m => m.id), sessionIdRef.current);
      }
      return prev.slice(0, start);
    });
  }, []);

  const cancelAllPolls = useCallback(() => {
    for (const { controller, wake } of pollControllersRef.values()) {
      controller.abort();
      wake();
    }
    pollControllersRef.clear();
  }, [pollControllersRef]);

  const pollTask = useCallback(async (taskId: string) => {
    const previous = pollControllersRef.get(taskId);
    previous?.controller.abort();
    previous?.wake();

    const controller = new AbortController();
    let wake = () => {};
    const pollState = { controller, wake };
    pollControllersRef.set(taskId, pollState);

    try {
      for (;;) {
        if (controller.signal.aborted || !mountedRef.current) throw new Error('已停止');
        const task = await getNovaTask(taskId);
        if (controller.signal.aborted || !mountedRef.current) throw new Error('已停止');
        if (task.status === 'completed') return task;
        if (task.status === 'failed' || task.status === 'expired') {
          throw new Error(task.error || task.warning || '生图任务失败');
        }
        await new Promise<void>(resolve => {
          const onAbort = () => {
            clearTimeout(timer);
            controller.signal.removeEventListener('abort', onAbort);
            resolve();
          };
          const timer = setTimeout(() => {
            controller.signal.removeEventListener('abort', onAbort);
            resolve();
          }, 4000);
          wake = onAbort;
          pollState.wake = onAbort;
          if (controller.signal.aborted) onAbort();
          else controller.signal.addEventListener('abort', onAbort, { once: true });
        });
      }
    } finally {
      if (pollControllersRef.get(taskId)?.controller === controller) {
        pollControllersRef.delete(taskId);
      }
    }
  }, [pollControllersRef]);

  const pendingGenerationWriteRef = useRef(Promise.resolve());
  const queuePendingGenerationWrite = useCallback((write: () => Promise<void>) => {
    const next = pendingGenerationWriteRef.current.then(write, write);
    pendingGenerationWriteRef.current = next.catch(() => {});
    return next;
  }, []);

  const persistPendingGeneration = useCallback((data: PendingGenerationData) => {
    pendingGenerationTaskRef.current = data.taskId;
    return queuePendingGenerationWrite(() => savePendingGeneration(data, sessionIdRef.current));
  }, [queuePendingGenerationWrite]);

  const clearPendingGenerationForTask = useCallback(async (taskId?: string) => {
    if (!mountedRef.current) return;
    if (taskId && pendingGenerationTaskRef.current !== taskId) return;
    pendingGenerationTaskRef.current = null;
    await queuePendingGenerationWrite(() => clearPendingGeneration(sessionIdRef.current));
  }, [queuePendingGenerationWrite]);

  /**
   * 生图任务完成后的统一后处理：下载图片 → 缩略图 + 视觉描述 → 登记 →
   * 合并成一条助手消息 → 清理生图状态。approveProposal 与 resumeGeneration
   * 此前各自重复了这段约 100 行逻辑，这里抽成单一实现，差异通过 ctx 注入。
   */
  const processGeneratedTask = useCallback(async (
    allImages: string[],
    ctx: {
      taskId: string;
      prompt: string;
      analysisFallbackReason: string;
      proposalData: AgentMessage['proposalData'];
    },
    options?: { background?: boolean },
  ): Promise<void> => {
    if (!mountedRef.current) return;
    const background = options?.background === true;
    const descController = new AbortController();
    describeAbortRef.current = descController;

    // 先下载所有图片（后台任务不改 phase，避免顶掉正在展示的下一个提案）
    if (!background) setPhase('loading');
    const blobs = await Promise.allSettled(allImages.map(ref => resultImageToBlob(ref)));
    if (!mountedRef.current) return;

    // 再生成缩略图 + 视觉描述
    if (!background) setPhase('describing');
    const records: AgentImageRecord[] = [];
    const errors: string[] = [];
    try {
      for (let i = 0; i < allImages.length; i++) {
        if (!mountedRef.current) return;
        try {
          const settled = blobs[i];
          const blob = settled && settled.status === 'fulfilled' ? settled.value : null;
          if (!blob) { errors.push('图片下载失败'); continue; }
          const preview = await makePreviewFromBlob(blob);
          if (!mountedRef.current) return;
          const record = await ingestImage('generated', blob, preview.dataUrl, blob.type || 'image/png', ctx.taskId, { width: preview.width, height: preview.height }, undefined, descController.signal);
          if (!mountedRef.current) return;
          records.push(record);
        } catch (err) {
          if (!mountedRef.current) return;
          errors.push(err instanceof Error ? err.message : String(err));
        }
      }
    } finally {
      if (describeAbortRef.current === descController) {
        describeAbortRef.current = null;
      }
    }

    if (!mountedRef.current || records.length === 0) {
      throw new Error(errors[0] || '图片处理失败');
    }

    const imgIds = records.map(r => r.imgId);
    const imgList = imgIds.join('、');
    // 后台并发任务不消费共享分析文本（属于正在展示的提案），只用任务自带上下文
    const analysis = background ? (ctx.analysisFallbackReason || '') : (pendingAnalysisRef.current || ctx.analysisFallbackReason || '');
    const reasoning = background ? '' : pendingReasoningRef.current;
    pendingAnalysisRef.current = background ? pendingAnalysisRef.current : '';
    pendingReasoningRef.current = background ? pendingReasoningRef.current : '';
    let generatedText = '';
    generatedText += `分析：${analysis || '根据你的描述，已为你生成图片。'}\n`;
    generatedText += `优化提示词：${ctx.prompt}\n`;
    generatedText += `结果：已生成图片 ${imgList}。需要继续调整就告诉我。`;
    if (errors.length > 0) {
      generatedText += `\n（部分图片处理失败：${errors.join('；')}）`;
    }

    appendMessage({
      id: generateUUID(),
      role: 'assistant',
      text: generatedText,
      reasoning: reasoning || undefined,
      imageIds: imgIds,
      taskId: ctx.taskId,
      proposalData: ctx.proposalData,
      createdAt: Date.now(),
    });

    // 后台并发任务（多商品批量批准）：不动 phase/提案/生成状态、不消费共享的分析文本；
    // 仅在面板空闲时让模型检查批量轮是否还有没做完的。
    if (background) {
      void clearPendingGenerationForTask(ctx.taskId);
      if (phaseRef.current === 'idle') maybeAutoContinue();
      return;
    }

    void clearPendingGenerationForTask(ctx.taskId);
    setGeneratingTaskId(null);
    setGeneratingStartedAt(null);
    setGenerationDraft(null);
    setIsSyncing(false);
    // 批量商品流程：当前商品生成完成后自动推进到队列中的下一个提案。
    // 读写都在 setState updater 之外，避免 StrictMode 双调用导致消息/持久化重复。
    const [next, ...rest] = proposalQueueRef.current;
    if (!next) {
      // 队列已空：批量轮里让模型检查是否还有没做完的；没有（或普通单图轮）就到 idle
      if (!maybeAutoContinue()) setPhase('idle');
      return;
    }
    proposalQueueRef.current = rest;
    setProposalQueue(rest);
    setProposal(next);
    setPhase('proposal');
    pendingAnalysisRef.current = '';
    pendingReasoningRef.current = '';
    isReeditRef.current = false;
    if (mountedRef.current) {
      void savePendingProposal(
        { proposal: next, pendingAnalysis: '', pendingReasoning: '', isReedit: false, queuedProposals: rest },
        sessionIdRef.current,
      );
    }
    appendMessage({
      id: generateUUID(),
      role: 'system-note',
      text: `当前商品已生成，继续处理下一个商品的提案（剩余 ${rest.length} 个待确认）。`,
      createdAt: Date.now(),
    });
  }, [appendMessage, clearPendingGenerationForTask, ingestImage, maybeAutoContinue]);

  /**
   * 后台并发生图任务：多商品批量批准时，非末尾提案的生成转后台轮询，
   * 完成后只登记图片+落结果消息，不占用前台的生成进度视图。
   */
  const trackBackgroundTask = useCallback(async (
    taskId: string,
    ctx: {
      taskId: string;
      prompt: string;
      analysisFallbackReason: string;
      proposalData: AgentMessage['proposalData'];
    },
  ): Promise<void> => {
    try {
      const task = await pollTask(taskId);
      if (!mountedRef.current) return;
      const allImages = task.result?.images;
      if (!allImages || allImages.length === 0) throw new Error('后端未返回图片');
      await processGeneratedTask(allImages, ctx, { background: true });
    } catch (err) {
      void clearPendingGenerationForTask(taskId);
      if (!mountedRef.current || isStoppedError(err)) return;
      appendMessage({
        id: generateUUID(),
        role: 'system-note',
        text: `一个商品的图片生成失败：${err instanceof Error ? err.message : String(err)}。其他商品不受影响。`,
        createdAt: Date.now(),
      });
    }
  }, [appendMessage, clearPendingGenerationForTask, pollTask, processGeneratedTask]);

  /** 页面刷新后恢复生图轮询：使用持久化的 generation 数据继续轮询并处理结果 */
  const resumeGeneration = useCallback(async (data: PendingGenerationData) => {
    try {
      const task = await pollTask(data.taskId);
      if (!mountedRef.current) return;
      const allImages = task.result?.images;
      if (!allImages || allImages.length === 0) throw new Error('后端未返回图片');

      await processGeneratedTask(allImages, {
        taskId: data.taskId,
        prompt: data.proposal.prompt,
        analysisFallbackReason: data.proposal.reason || '',
        proposalData: {
          action: data.selectedImageIds.length > 0 ? 'edit' : 'generate',
          prompt: data.proposal.prompt,
          referencedImageIds: data.selectedImageIds,
          model: data.model as ModelId,
          outputSize: data.outputSize,
          customSize: data.customSize,
          aspectRatio: data.aspectRatio,
          temperature: data.temperature,
          gptImageQuality: data.gptImageQuality,
          gptImageStyle: data.gptImageStyle,
          gptImageBackground: data.gptImageBackground,
          parallelCount: data.parallelCount,
          productKey: data.proposal.productKey,
          productName: data.proposal.productName,
        },
      });
    } catch (err) {
      void clearPendingGenerationForTask(data.taskId);
      if (!mountedRef.current || isStoppedError(err)) return;
      setError(err instanceof Error ? err.message : '生图失败');
      setProposal({
        action: data.proposal?.action ?? (data.selectedImageIds.length > 0 ? 'edit' : 'generate'),
        prompt: data.proposal.prompt,
        referencedImageIds: data.selectedImageIds,
        reason: data.proposal?.reason ?? '',
        productKey: data.proposal?.productKey,
        productName: data.proposal?.productName,
        requestedAspectRatio: data.proposal?.requestedAspectRatio,
        suggestedAspectRatio: data.proposal?.suggestedAspectRatio ?? data.aspectRatio,
        requestedOutputSize: data.proposal?.requestedOutputSize ?? data.outputSize,
        temperature: data.temperature,
        gptImageQuality: data.gptImageQuality,
        gptImageStyle: data.gptImageStyle,
        gptImageBackground: data.gptImageBackground,
        parallelCount: data.parallelCount,
      });
      setGeneratingTaskId(null);
      setGeneratingStartedAt(null);
      setGenerationDraft(null);
      setIsSyncing(false);
      setPhase('proposal');
    }
  }, [clearPendingGenerationForTask, pollTask, processGeneratedTask]);

  useEffect(() => {
    if (!ready || !generationToResumeRef.current) return;
    const generation = generationToResumeRef.current;
    generationToResumeRef.current = null;
    void resumeGeneration(generation).catch(() => {});
  }, [ready, resumeGeneration]);

  const checkNow = useCallback(async (): Promise<AgentCheckResult> => {
    if (phase !== 'generating') return 'idle';
    const taskId = generatingTaskId;
    if (!taskId) return 'idle';

    setIsSyncing(true);
    // 立即唤醒当前前台任务的轮询，让完成/失败的状态切换尽快走正常流程
    pollControllersRef.get(taskId)?.wake();

    try {
      const task = await getNovaTask(taskId);
      if (task.status === 'completed') return 'completed';
      if (task.status === 'failed' || task.status === 'expired') return 'failed';
      if (task.status === 'processing') return 'processing';
      return 'queued';
    } catch {
      return 'error';
    } finally {
      setIsSyncing(false);
    }
  }, [generatingTaskId, phase, pollControllersRef]);

  const approveProposal = useCallback(async (
    finalPrompt: string,
    selectedImageIds: string[],
    model: string,
    params: AgentResolvedLayout,
  ) => {
    if (phase !== 'proposal') return;
    const generationEpoch = generationEpochRef.current;
    const prompt = finalPrompt.trim();
    if (prompt.length === 0) {
      setError('提示词不能为空');
      return;
    }
    setError(null);
    const startedAt = Date.now();
    const approvedProposal: AgentProposal = {
      action: proposal?.action ?? (selectedImageIds.length > 0 ? 'edit' : 'generate'),
      prompt,
      referencedImageIds: selectedImageIds,
      reason: proposal?.reason ?? '',
      productKey: proposal?.productKey,
      productName: proposal?.productName,
      requestedAspectRatio: proposal?.requestedAspectRatio,
      suggestedAspectRatio: proposal?.suggestedAspectRatio ?? params.aspectRatio,
      requestedOutputSize: proposal?.requestedOutputSize ?? params.outputSize,
      temperature: params.temperature,
      gptImageQuality: params.gptImageQuality,
      gptImageStyle: params.gptImageStyle,
      gptImageBackground: params.gptImageBackground,
      parallelCount: params.parallelCount,
      requestedModelId: proposal?.requestedModelId,
    };
    proposalRef.current = approvedProposal;
    setProposal(null);
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }
    setPhase('generating');
    setGeneratingStartedAt(startedAt);
    setGenerationDraft({
      analysis: pendingAnalysisRef.current || approvedProposal.reason || '根据你的描述，正在生成图片。',
      reasoning: pendingReasoningRef.current || undefined,
      prompt,
      parallelCount: params.parallelCount,
      startedAt,
    });

    let createdTaskId: string | null = null;
    try {
      const references: ImageReference[] = [];
      for (const imgId of selectedImageIds) {
        if (!mountedRef.current || generationEpochRef.current !== generationEpoch) return;
        const bytes = await getAgentImageBase64(imgId, sessionIdRef.current);
        if (!mountedRef.current || generationEpochRef.current !== generationEpoch) return;
        if (bytes) references.push({ data: bytes.data, mimeType: bytes.mimeType });
      }
      const mode = references.length > 0 ? 'image-to-image' : 'text-to-image';
      const provider = resolveImageTaskProvider(model);
      if (!mountedRef.current || generationEpochRef.current !== generationEpoch) return;
      const layout = resolveSubmitLayout(model, params.outputSize, params.aspectRatio, prompt);

      const taskId = await createNovaTask({
        apiKey: provider.apiKey,
        baseUrl: provider.baseUrl,
        protocol: provider.protocol,
        mode,
        prompt,
        outputSize: layout.outputSize,
        customSize: layout.outputSize === 'auto' ? undefined : params.customSize,
        aspectRatio: layout.aspectRatio,
        temperature: params.temperature,
        model: provider.modelId,
        gptImageQuality: params.gptImageQuality,
        gptImageStyle: params.gptImageStyle,
        gptImageBackground: params.gptImageBackground,
        parallelCount: params.parallelCount,
        images: references,
      });
      createdTaskId = taskId;
      if (!mountedRef.current || generationEpochRef.current !== generationEpoch) return;
      setGeneratingTaskId(taskId);
      setGenerationDraft(prev => prev ? { ...prev, taskId } : prev);
      const pendingGenerationData: PendingGenerationData = {
        taskId,
        proposal: approvedProposal,
        pendingAnalysis: pendingAnalysisRef.current,
        pendingReasoning: pendingReasoningRef.current,
        selectedImageIds,
        model,
        outputSize: layout.outputSize,
        customSize: layout.outputSize === 'auto' ? undefined : params.customSize,
        aspectRatio: layout.aspectRatio,
        temperature: params.temperature,
        gptImageQuality: params.gptImageQuality,
        gptImageStyle: params.gptImageStyle,
        gptImageBackground: params.gptImageBackground,
        parallelCount: params.parallelCount,
        startedAt,
      };

      // 队列里还有后续商品提案：本任务转后台并发，立刻展示下一个提案，实现「批准即提交、一次生成多张」
      // PendingGeneration 目前是单槽：只在没有前台/已保存任务时占用它，避免后台任务覆盖当前前台任务。
      const [nextProposal, ...restQueue] = proposalQueueRef.current;
      if (nextProposal) {
        proposalQueueRef.current = restQueue;
        setProposalQueue(restQueue);
        setProposal(nextProposal);
        setPhase('proposal');
        setGeneratingTaskId(null);
        setGeneratingStartedAt(null);
        setGenerationDraft(null);
        pendingAnalysisRef.current = '';
        pendingReasoningRef.current = '';
        if (mountedRef.current) {
          void savePendingProposal(
            { proposal: nextProposal, pendingAnalysis: '', pendingReasoning: '', isReedit: false, queuedProposals: restQueue },
            sessionIdRef.current,
          );
          if (pendingGenerationTaskRef.current === null) {
            void persistPendingGeneration(pendingGenerationData);
          }
        }
        appendMessage({
          id: generateUUID(),
          role: 'system-note',
          text: `《${approvedProposal.productName || '当前商品'}》已提交生成（后台并发），继续确认下一个商品的提案（剩余 ${restQueue.length} 个）。`,
          createdAt: Date.now(),
        });
        void trackBackgroundTask(taskId, {
          taskId,
          prompt,
          analysisFallbackReason: approvedProposal.reason || '',
          proposalData: {
            action: selectedImageIds.length > 0 ? 'edit' : 'generate',
            prompt,
            referencedImageIds: selectedImageIds,
            model,
            outputSize: layout.outputSize,
            customSize: layout.outputSize === 'auto' ? undefined : params.customSize,
            aspectRatio: layout.aspectRatio,
            temperature: params.temperature,
            gptImageQuality: params.gptImageQuality,
            gptImageStyle: params.gptImageStyle,
            gptImageBackground: params.gptImageBackground,
            parallelCount: params.parallelCount,
            productKey: approvedProposal.productKey,
            productName: approvedProposal.productName,
          },
        });
        return;
      }

      if (mountedRef.current) {
        void persistPendingGeneration(pendingGenerationData);
      }

      const task = await pollTask(taskId);
      if (!mountedRef.current) return;
      const allImages = task.result?.images;
      if (!allImages || allImages.length === 0) throw new Error('后端未返回图片');

      await processGeneratedTask(allImages, {
        taskId,
        prompt,
        analysisFallbackReason: proposalRef.current?.reason || '',
        proposalData: {
          action: selectedImageIds.length > 0 ? 'edit' : 'generate',
          prompt,
          referencedImageIds: selectedImageIds,
          model,
          outputSize: layout.outputSize,
          customSize: layout.outputSize === 'auto' ? undefined : params.customSize,
          aspectRatio: layout.aspectRatio,
          temperature: params.temperature,
          gptImageQuality: params.gptImageQuality,
          gptImageStyle: params.gptImageStyle,
          gptImageBackground: params.gptImageBackground,
          parallelCount: params.parallelCount,
          productKey: approvedProposal.productKey,
          productName: approvedProposal.productName,
        },
      });
    } catch (err) {
      if (createdTaskId) void clearPendingGenerationForTask(createdTaskId);
      if (!mountedRef.current || generationEpochRef.current !== generationEpoch || isStoppedError(err)) return;
      setError(err instanceof Error ? err.message : '生图失败');
      setProposal({
        action: approvedProposal.action,
        prompt,
        referencedImageIds: selectedImageIds,
        reason: approvedProposal.reason,
        productKey: approvedProposal.productKey,
        productName: approvedProposal.productName,
        requestedAspectRatio: approvedProposal.requestedAspectRatio,
        suggestedAspectRatio: approvedProposal.suggestedAspectRatio,
        requestedOutputSize: approvedProposal.requestedOutputSize,
        temperature: params.temperature,
        gptImageQuality: params.gptImageQuality,
        gptImageStyle: params.gptImageStyle,
        gptImageBackground: params.gptImageBackground,
        parallelCount: params.parallelCount,
        requestedModelId: approvedProposal.requestedModelId,
      });
      setGeneratingTaskId(null);
      setGeneratingStartedAt(null);
      setGenerationDraft(null);
      setIsSyncing(false);
      setPhase('proposal');
    }
  }, [appendMessage, clearPendingGenerationForTask, persistPendingGeneration, phase, proposal, pollTask, processGeneratedTask, trackBackgroundTask]);

  const stopStreaming = useCallback(() => {
    generationEpochRef.current += 1;
    streamHandleRef.current?.abort();
    streamHandleRef.current = null;
    cancelAllPolls();
    flushAndCancelRaf();
    setStreamingText('');
    setStreamingReasoning('');
    setGeneratingTaskId(null);
    setGeneratingStartedAt(null);
    setGenerationDraft(null);
    setIsSyncing(false);
    setPhase('idle');
    describeAbortRef.current?.abort();
    // 用户主动停止 = 批量轮一并作废：清掉链接轮状态与待确认队列，
    // 否则后续任务完成回调里的 maybeAutoContinue 还会在用户喊停后自动续跑。
    userLinksRef.current = [];
    autoContinueCountRef.current = 0;
    proposalQueueRef.current = [];
    setProposalQueue([]);
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }
    void clearPendingGenerationForTask();
  }, [cancelAllPolls, clearPendingGenerationForTask, flushAndCancelRaf]);

  const skipDescribing = useCallback(() => {
    describeAbortRef.current?.abort();
    describeAbortRef.current = null;
  }, []);

  const setImageModel = useCallback((model: ModelId) => {
    if (!mountedRef.current) return;
    setImageModelState(model);
    void saveImageModel(model, sessionIdRef.current);
  }, []);

  const toggleWebSearch = useCallback(() => {
    if (!agentSupportsWebSearch()) return;
    setWebSearchEnabled(prev => {
      const next = !prev;
      try { localStorage.setItem('nova-agent-web-search', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, [agentSupportsWebSearch]);

  const toggleCdp = useCallback(() => {
    setCdpEnabled(prev => {
      const next = !prev;
      try { localStorage.setItem('nova-agent-cdp', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const toggleIntentRecognition = useCallback(() => {
    setIntentRecognition(prev => {
      const next = !prev;
      try { localStorage.setItem('nova-agent-intent-recognition', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  // 清理上下文：插入一个分隔点，分隔点之前的对话与图片对模型不再可见，但界面保留可见。
  const clearContext = useCallback(() => {
    if (!mountedRef.current || phase !== 'idle') return;
    setMessages(prev => {
      const lastReal = [...prev].reverse().find(m => m.role !== 'context-divider');
      if (!lastReal) return prev;
      if (prev[prev.length - 1]?.role === 'context-divider') return prev;
      const divider: AgentMessage = {
        id: generateUUID(),
        role: 'context-divider',
        text: '以下为新对话，助手已不记得上文',
        createdAt: Date.now(),
      };
      if (mountedRef.current) {
        void putMessage(divider, sessionIdRef.current);
      }
      return [...prev, divider];
    });
    setProposal(null);
    setError(null);
    // 上下文已分隔，模型不再记得之前的链接与批量轮；批量轮状态必须同步忘掉，
    // 否则 maybeAutoContinue 会拿模型看不见的链接继续续跑。
    userLinksRef.current = [];
    autoContinueCountRef.current = 0;
    proposalQueueRef.current = [];
    setProposalQueue([]);
  }, [phase]);

  const clearSession = useCallback(async () => {
    generationEpochRef.current += 1;
    streamHandleRef.current?.abort();
    streamHandleRef.current = null;
    cancelAllPolls();
    describeAbortRef.current?.abort();
    await clearPendingGenerationForTask();
    await clearAgentSession(sessionIdRef.current);
    if (!mountedRef.current) return;
    setMessages([]);
    setImages([]);
    setProposal(null);
    // 全量重开：消息/图片/批量轮的 ref 与 state 一起归零，不能等 useEffect 滞后同步
    messagesRef.current = [];
    imagesRef.current = [];
    proposalRef.current = null;
    userLinksRef.current = [];
    autoContinueCountRef.current = 0;
    proposalQueueRef.current = [];
    setProposalQueue([]);
    flushAndCancelRaf();
    setStreamingText('');
    setStreamingReasoning('');
    setGeneratingTaskId(null);
    setGeneratingStartedAt(null);
    setGenerationDraft(null);
    setIsSyncing(false);
    setError(null);
    seqRef.current = 0;
    setPhase('idle');
  }, [cancelAllPolls, clearPendingGenerationForTask, flushAndCancelRaf]);

  /** 根据消息中的 proposalData 重新打开提案编辑 */
  const reeditProposal = useCallback((messageId: string) => {
    // 仅在空闲时允许重建提案：生成中/提案确认中重入会顶掉当前提案与生成状态
    if (phaseRef.current !== 'idle') return;
    const message = messages.find(m => m.id === messageId);
    if (!message?.proposalData) return;
    const pd = message.proposalData;
    const advancedParams = getGptImageAdvancedParamsForModel(pd.model, {
      quality: pd.gptImageQuality,
      style: pd.gptImageStyle,
      background: pd.gptImageBackground,
    });
    // 构建 AgentProposal 重新进入 proposal 阶段
    const newProposal: AgentProposal = {
      action: pd.action,
      prompt: pd.prompt,
      referencedImageIds: pd.referencedImageIds,
      reason: '重新编辑之前的生图请求。',
      requestedAspectRatio: undefined,
      suggestedAspectRatio: pd.aspectRatio,
      requestedOutputSize: pd.outputSize,
      temperature: pd.temperature,
      gptImageQuality: advancedParams.quality,
      gptImageStyle: advancedParams.style,
      gptImageBackground: advancedParams.background,
      parallelCount: pd.parallelCount,
      // 重新编辑必须保留商品作用域，否则 scopeAgentProposal 无法过滤，
      // 参考图会混入其他商品的图（串图）
      productKey: pd.productKey,
      productName: pd.productName,
      requestedModelId: pd.model,
    };
    // 重新编辑时恢复原始生图模型
    const reeditCatalog = buildModelCatalog();
    const resolvedModel = resolveValidAgentImageModel(resolveAgentModel(
      imageModelRef.current,
      newProposal.requestedModelId,
      newProposal.requestedOutputSize,
      reeditCatalog,
    ));
    if (resolvedModel !== imageModelRef.current) {
      imageModelRef.current = resolvedModel;
      setImageModelState(resolvedModel);
      if (mountedRef.current) {
                void saveImageModel(resolvedModel, sessionIdRef.current);
              }
    }
    // 清除上次待定分析，因为用户要重新编辑
    pendingAnalysisRef.current = '';
    pendingReasoningRef.current = '';
    isReeditRef.current = true;
    setProposal(newProposal);
    setPhase('proposal');
    if (mountedRef.current) {
      void savePendingProposal({
        proposal: newProposal,
        pendingAnalysis: '',
        pendingReasoning: '',
        isReedit: true,
      }, sessionIdRef.current);
    }
  }, [messages]);

  /** 清理指定消息引用的且不再被其他消息使用的图片 */
  const cleanupOrphanImages = useCallback((keptMessages: AgentMessage[], removedImageIds: string[]) => {
    const uniqueIds = [...new Set(removedImageIds)];
    for (const imgId of uniqueIds) {
      const stillReferenced = keptMessages.some(m => m.imageIds?.includes(imgId));
      if (!stillReferenced) {
        setImages(prev => prev.filter(img => img.imgId !== imgId));
        void deleteImageRecords([imgId], sessionIdRef.current);
        void deleteAgentImageBytes(imgId, sessionIdRef.current);
      }
    }
  }, []);

  /** 删除单条消息（用户或助手），同时清理关联的图片资源 */
  const deleteMessage = useCallback((messageId: string) => {
    const message = messages.find(m => m.id === messageId);
    if (!message) return;
    const removedImageIds = message.imageIds || [];
    setMessages(prev => prev.filter(m => m.id !== messageId));
    if (mountedRef.current) {
      void deleteMessages([messageId], sessionIdRef.current);
    }
    cleanupOrphanImages(messages.filter(m => m.id !== messageId), removedImageIds);
  }, [messages, cleanupOrphanImages]);

  /** 撤回：删除从指定消息开始（含）之后的所有消息，同时清理关联图片 */
  const rollbackMessages = useCallback((fromMessageId: string) => {
    const fromIndex = messages.findIndex(m => m.id === fromMessageId);
    if (fromIndex === -1) return;
    const toRemove = messages.slice(fromIndex);
    const removedImageIds = toRemove.flatMap(m => m.imageIds || []);
    setMessages(prev => prev.slice(0, fromIndex));
    if (mountedRef.current) {
      void deleteMessages(toRemove.map(m => m.id), sessionIdRef.current);
    }
    cleanupOrphanImages(messages.slice(0, fromIndex), removedImageIds);
    // 如果当前在 proposal 阶段且涉及被删除的上下文，重置
    setProposal(null);
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }
    flushAndCancelRaf();
    setStreamingText('');
    setStreamingReasoning('');
    if (phase !== 'idle') setPhase('idle');
  }, [messages, phase, cleanupOrphanImages, flushAndCancelRaf]);

  /**
   * 最后一条用户消息的 id；不存在或其后还有别的用户消息时为 null。
   *
   * 重试**只允许**作用于它，这是刻意的限制：重试中间某轮意味着要丢弃它之后
   * 的全部对话，否则模型会看到「同一个问题两个不同答案」的历史而错乱。
   * 与其偷偷替用户删掉后面几轮，不如只在最后一轮给出重试入口 ——
   * 想改中间某轮，用现成的「撤回以下所有」把尾巴清掉再重试。
   */
  const retryableMessageId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === 'user') return message.id;
      // 分隔线之后没有用户消息 → 没有可重试的一轮
      if (message.role === 'context-divider') return null;
    }
    return null;
  }, [messages]);

  /**
   * 重试最后一条用户消息：删掉它之后的助手回复，用同一条用户消息重新发起请求。
   *
   * 用户消息本身**保留**（不重新登记图片、不重算描述），只回滚模型侧产物。
   * 因此关联图片一律不清理 —— 它们仍被那条保留下来的用户消息引用。
   */
  const retryMessage = useCallback((messageId: string) => {
    if (phase !== 'idle') return;
    const index = messages.findIndex(m => m.id === messageId);
    if (index === -1) return;
    const target = messages[index];
    if (target.role !== 'user' || messageId !== retryableMessageId) return;

    // 丢弃这条用户消息之后的所有内容（助手回复、系统提示、提案分析）
    const toRemove = messages.slice(index + 1);
    const kept = messages.slice(0, index + 1);
    if (toRemove.length > 0) {
      setMessages(kept);
      if (mountedRef.current) {
        void deleteMessages(toRemove.map(m => m.id), sessionIdRef.current);
      }
      // 只清理「被删除消息引用、且保留部分不再引用」的图片。
      // 用户消息还在，它引用的上传图不会被误删。
      cleanupOrphanImages(kept, toRemove.flatMap(m => m.imageIds || []));
    }

    pendingAnalysisRef.current = '';
    pendingReasoningRef.current = '';
    isReeditRef.current = false;
    setProposal(null);
    if (mountedRef.current) {
      void clearPendingProposal(sessionIdRef.current);
    }
    flushAndCancelRaf();
    setStreamingText('');
    setStreamingReasoning('');
    setError(null);

    const { history, catalog } = sliceActiveContext(kept, images);
    runChat(history, catalog);
  }, [phase, messages, retryableMessageId, images, cleanupOrphanImages, flushAndCancelRaf, runChat]);

  // 组件卸载时清理：取消 rAF + 停止轮询/流式/描述，避免卸载后仍每 4s 轮询、
  // 在卸载后继续下载/写库/setState（内存泄漏 + 卸载后写状态）。
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cancelAllPolls();
      streamHandleRef.current?.abort();
      describeAbortRef.current?.abort();
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [cancelAllPolls]);

  return {
    ready,
    hasApiKey,
    phase,
    messages,
    images,
    proposal,
    streamingText,
    streamingReasoning,
    imageModel,
    error,
    generatingTaskId,
    generatingStartedAt,
    generationDraft,
    isSyncing,
    webSearchEnabled,
    agentSupportsWebSearch: agentSupportsWebSearch(),
    cdpEnabled,
    intentRecognition,
    sendMessage,
    approveProposal,
    cancelProposal,
    reeditProposal,
    withdrawTurn,
    deleteMessage,
    rollbackMessages,
    retryMessage,
    retryableMessageId,
    checkNow,
    stopStreaming,
    skipDescribing,
    setImageModel,
    toggleWebSearch,
    toggleCdp,
    toggleIntentRecognition,
    clearSession,
    clearContext,
    redescribeImage,
    dismissError: () => setError(null),
  };
}
