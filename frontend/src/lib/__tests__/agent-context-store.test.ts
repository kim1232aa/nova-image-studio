import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type FakeKey = string | number;

type FakeState = {
  name: string;
  stores: Map<string, Map<FakeKey, unknown>>;
  connections: Set<FakeDatabase>;
};

class FakeRequest<T = unknown> {
  result!: T;
  error: DOMException | null = null;
  onsuccess: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onupgradeneeded: ((event: Event) => void) | null = null;
  onblocked: ((event: Event) => void) | null = null;
}

class FakeTransaction {
  oncomplete: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  private pending = 0;

  constructor(private readonly database: FakeDatabase, private readonly storeNames: string[]) {}

  objectStore(name: string): FakeObjectStore {
    if (!this.storeNames.includes(name)) throw new Error(`Store not in transaction: ${name}`);
    return new FakeObjectStore(this, this.database.state, name);
  }

  enqueue<T>(request: FakeRequest<T>, operation: () => T): FakeRequest<T> {
    this.pending += 1;
    queueMicrotask(() => {
      try {
        request.result = operation();
        request.onsuccess?.({ target: request } as unknown as Event);
      } catch (error) {
        request.error = error as DOMException;
        request.onerror?.({ target: request } as unknown as Event);
      } finally {
        this.pending -= 1;
        if (this.pending === 0) {
          queueMicrotask(() => this.oncomplete?.({ target: this } as unknown as Event));
        }
      }
    });
    return request;
  }
}

class FakeObjectStore {
  constructor(
    private readonly transaction: FakeTransaction,
    private readonly state: FakeState,
    private readonly name: string,
  ) {}

  getAll(): FakeRequest<unknown[]> {
    const request = new FakeRequest<unknown[]>();
    return this.transaction.enqueue(request, () => Array.from(this.store().values()));
  }

  get(key: FakeKey): FakeRequest<unknown> {
    const request = new FakeRequest<unknown>();
    return this.transaction.enqueue(request, () => this.store().get(key));
  }

  put(value: unknown): FakeRequest<FakeKey> {
    const request = new FakeRequest<FakeKey>();
    return this.transaction.enqueue(request, () => {
      const keyPath = this.name === 'messages' ? 'id' : this.name === 'images' ? 'imgId' : 'key';
      const key = (value as Record<string, unknown>)[keyPath];
      if (typeof key !== 'string' && typeof key !== 'number') {
        throw new Error(`Missing key path: ${keyPath}`);
      }
      this.store().set(key, value);
      return key;
    });
  }

  delete(key: FakeKey): FakeRequest<undefined> {
    const request = new FakeRequest<undefined>();
    return this.transaction.enqueue(request, () => {
      this.store().delete(key);
      return undefined;
    });
  }

  clear(): FakeRequest<undefined> {
    const request = new FakeRequest<undefined>();
    return this.transaction.enqueue(request, () => {
      this.store().clear();
      return undefined;
    });
  }

  private store(): Map<FakeKey, unknown> {
    const store = this.state.stores.get(this.name);
    if (!store) throw new Error(`Unknown store: ${this.name}`);
    return store;
  }
}

class FakeDatabase {
  readonly objectStoreNames = {
    contains: (name: string) => this.state.stores.has(name),
  } as unknown as DOMStringList;
  onversionchange: ((event: IDBVersionChangeEvent) => void) | null = null;
  onclose: ((event: Event) => void) | null = null;
  closed = false;
  closeCount = 0;

  constructor(readonly state: FakeState) {
    state.connections.add(this);
  }

  createObjectStore(name: string): void {
    this.state.stores.set(name, new Map());
  }

  transaction(storeNames: string | string[]): FakeTransaction {
    return new FakeTransaction(this, Array.isArray(storeNames) ? storeNames : [storeNames]);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.closeCount += 1;
    this.state.connections.delete(this);
  }
}

function createFakeIndexedDB() {
  const states = new Map<string, FakeState>();
  const openCalls: string[] = [];
  const deleteCalls: string[] = [];

  const factory = {
    open(name: string): FakeRequest<IDBDatabase> {
      openCalls.push(name);
      const request = new FakeRequest<IDBDatabase>();
      queueMicrotask(() => {
        let state = states.get(name);
        const isNew = !state;
        if (!state) {
          state = { name, stores: new Map(), connections: new Set() };
          states.set(name, state);
        }
        const database = new FakeDatabase(state);
        request.result = database as unknown as IDBDatabase;
        if (isNew) request.onupgradeneeded?.({ target: request } as unknown as Event);
        request.onsuccess?.({ target: request } as unknown as Event);
      });
      return request;
    },

    deleteDatabase(name: string): FakeRequest<undefined> {
      deleteCalls.push(name);
      const request = new FakeRequest<undefined>();
      queueMicrotask(() => {
        const state = states.get(name);
        if (state && state.connections.size > 0) {
          request.onblocked?.({ target: request } as unknown as Event);
          request.error = new DOMException('blocked', 'InvalidStateError');
          request.onerror?.({ target: request } as unknown as Event);
          return;
        }
        states.delete(name);
        request.onsuccess?.({ target: request } as unknown as Event);
      });
      return request;
    },
  } as unknown as IDBFactory;

  return { factory, openCalls, deleteCalls, states };
}

function message(id: string, createdAt: number) {
  return { id, role: 'user' as const, text: id, createdAt };
}

function imageRecord(imgId: string, createdAt: number) {
  return {
    imgId,
    source: 'generated' as const,
    thumbnail: '',
    description: imgId,
    mimeType: 'text/plain',
    createdAt,
  };
}

function readBlobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

describe('agent-context-store session databases', () => {
  let store: typeof import('@/lib/agent-context-store');
  let fakeIndexedDB: ReturnType<typeof createFakeIndexedDB>;

  beforeEach(async () => {
    vi.resetModules();
    fakeIndexedDB = createFakeIndexedDB();
    vi.stubGlobal('indexedDB', fakeIndexedDB.factory);
    store = await import('@/lib/agent-context-store');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('默认会话沿用 nova-agent-db，切换后现有 API 按会话隔离', async () => {
    await store.putMessage(message('default-message', 1));
    expect(fakeIndexedDB.openCalls).toEqual(['nova-agent-db']);

    store.setAgentSession('review');
    expect((await store.loadAgentSession()).messages).toEqual([]);

    await store.putMessage(message('review-message', 2));
    store.setAgentSession('default');
    expect((await store.loadAgentSession()).messages.map(item => item.id)).toEqual(['default-message']);

    store.setAgentSession('review');
    expect((await store.loadAgentSession()).messages.map(item => item.id)).toEqual(['review-message']);
    expect(fakeIndexedDB.openCalls).toEqual(['nova-agent-db', 'nova-agent-db-review']);
  });

  it('删除非默认会话前关闭缓存句柄，并拒绝删除默认数据库', async () => {
    store.setAgentSession('review');
    await store.putMessage(message('review-message', 1));
    const reviewDb = Array.from(fakeIndexedDB.states.get('nova-agent-db-review')!.connections)[0];

    await store.deleteAgentSessionDatabase('review');

    expect(reviewDb.closeCount).toBe(1);
    expect(fakeIndexedDB.deleteCalls).toEqual(['nova-agent-db-review']);
    expect((await store.loadAgentSession()).messages).toEqual([]);

    await expect(store.deleteAgentSessionDatabase('default')).rejects.toThrow(/默认会话数据库不可删除/);
    expect(fakeIndexedDB.deleteCalls).toEqual(['nova-agent-db-review']);
  });

  it('延迟执行的旧会话写入仍固定到显式 sessionId', async () => {
    store.setAgentSession('old');
    const delayedWrite = Promise.resolve().then(() =>
      store.putMessage(message('old-message', 1), 'old'),
    );
    store.setAgentSession('new');

    await delayedWrite;

    expect((await store.loadAgentSession('old')).messages.map(item => item.id)).toEqual(['old-message']);
    expect((await store.loadAgentSession('new')).messages).toEqual([]);
  });

  it('非默认会话使用独立的 img blob 命名空间', async () => {
    await store.putImageRecord(imageRecord('img_1', 1), 'alpha');
    await store.storeAgentImageBytes('img_1', new Blob(['alpha'], { type: 'text/plain' }), 'alpha');
    await store.putImageRecord(imageRecord('img_1', 2), 'beta');
    await store.storeAgentImageBytes('img_1', new Blob(['beta'], { type: 'text/plain' }), 'beta');

    const alphaBlob = await store.getAgentImageBytes('img_1', 'alpha');
    const betaBlob = await store.getAgentImageBytes('img_1', 'beta');

    expect(alphaBlob).not.toBeNull();
    expect(betaBlob).not.toBeNull();
    expect(await readBlobText(alphaBlob!)).toBe('alpha');
    expect(await readBlobText(betaBlob!)).toBe('beta');
  });

  it('删除会话时清理其 blob，而不会影响另一个会话', async () => {
    await store.putImageRecord(imageRecord('img_1', 1), 'alpha');
    await store.storeAgentImageBytes('img_1', new Blob(['alpha'], { type: 'text/plain' }), 'alpha');
    await store.putImageRecord(imageRecord('img_1', 2), 'beta');
    await store.storeAgentImageBytes('img_1', new Blob(['beta'], { type: 'text/plain' }), 'beta');

    await store.deleteAgentSessionDatabase('alpha');

    expect(await store.getAgentImageBytes('img_1', 'alpha')).toBeNull();
    const betaBlob = await store.getAgentImageBytes('img_1', 'beta');
    expect(betaBlob).not.toBeNull();
    expect(await readBlobText(betaBlob!)).toBe('beta');
  });

  it('删除会话被阻塞时返回失败并保留原会话图片', async () => {
    await store.putMessage(message('blocked-message', 1), 'blocked');
    await store.putImageRecord(imageRecord('img_1', 1), 'blocked');
    await store.storeAgentImageBytes('img_1', new Blob(['blocked'], { type: 'text/plain' }), 'blocked');
    const extraRequest = fakeIndexedDB.factory.open('nova-agent-db-blocked');
    const extraConnection = await new Promise<FakeDatabase>((resolve) => {
      extraRequest.onsuccess = () => resolve(extraRequest.result as unknown as FakeDatabase);
    });

    await expect(store.deleteAgentSessionDatabase('blocked')).rejects.toThrow(/blocked/i);
    const retainedBlob = await store.getAgentImageBytes('img_1', 'blocked');
    expect(retainedBlob).not.toBeNull();
    expect(await readBlobText(retainedBlob!)).toBe('blocked');
    extraConnection.close();
  });
});
