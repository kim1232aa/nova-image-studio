import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import { StrictMode, type PropsWithChildren } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAgentChat } from '../useAgentChat';
import { AgentChatWorkspace } from '@/components/agent/AgentChatWorkspace';

const store = vi.hoisted(() => ({
  loadAgentSession: vi.fn(),
  loadPendingProposal: vi.fn(),
  loadPendingGeneration: vi.fn(),
  putMessage: vi.fn(),
  putImageRecord: vi.fn(),
  saveImageModel: vi.fn(),
  clearAgentSession: vi.fn(),
  storeAgentImageBytes: vi.fn(),
  getAgentImageBase64: vi.fn(),
  deleteMessages: vi.fn(),
  deleteImageRecords: vi.fn(),
  deleteAgentImageBytes: vi.fn(),
  savePendingProposal: vi.fn(),
  clearPendingProposal: vi.fn(),
  savePendingGeneration: vi.fn(),
  clearPendingGeneration: vi.fn(),
}));
const client = vi.hoisted(() => ({
  streamAgentChat: vi.fn(),
  describeImage: vi.fn(),
}));
const taskClient = vi.hoisted(() => ({
  createNovaTask: vi.fn(),
  getNovaTask: vi.fn(),
  resolveImageTaskProvider: vi.fn(),
}));
const imageDownloader = vi.hoisted(() => ({
  fetchImageAsBlob: vi.fn(),
}));

vi.mock('@/lib/agent-context-store', () => store);
vi.mock('@/lib/settings-storage', () => ({
  hasAnyApiKey: vi.fn(() => true),
  loadJsonFromStorage: vi.fn(() => ({})),
  saveJsonToStorage: vi.fn(),
}));
vi.mock('@/lib/uuid', () => ({ generateUUID: vi.fn(() => 'generated-message-id') }));
vi.mock('@/lib/model-capabilities', () => ({
  getGptImageAdvancedParamsForModel: vi.fn(() => ({ quality: 'auto', style: 'auto', background: 'auto' })),
  resolveAgentModel: vi.fn(),
  getAspectRatioOptions: vi.fn(() => [{ value: '1:1' }]),
  getCustomSizeMaxSide: vi.fn(() => 2048),
  getSupportsTemperature: vi.fn(() => false),
  getValidOutputSizes: vi.fn(() => ['1K']),
  normalizeCustomImageSize: vi.fn((value: string | undefined) => value),
  normalizeModel: vi.fn((value: string) => value),
  sanitizeLayoutForModel: vi.fn(),
  resolveSubmitLayout: vi.fn((_model: string, outputSize: string, aspectRatio: string) => ({ outputSize, aspectRatio })),
  supportsCustomSize: vi.fn(() => false),
  supportsGptImageAdvancedParams: vi.fn(() => false),
  PARALLEL_COUNT_VALUES: [1, 2, 3, 4],
  CUSTOM_IMAGE_SIZE_LIMITS: { multiple: 16, maxAspectRatio: 3, minPixels: 655360, maxPixels: 8294400 },
}));
vi.mock('@/lib/nova-models', () => ({
  getCompleteImageModels: vi.fn(() => []),
  getDefaultImageModel: vi.fn(() => undefined),
  getImageModelById: vi.fn(() => undefined),
  loadRegistry: vi.fn(() => ({})),
}));
vi.mock('@/lib/agent-chat-client', () => client);
vi.mock('@/lib/ccode-task-client', () => taskClient);
vi.mock('@/lib/image-downloader', () => imageDownloader);
vi.mock('@/lib/agent-cdp-tools', () => ({ executeAgentCdpTool: vi.fn() }));
vi.mock('@/lib/model-endpoints', () => ({
  getDefaultConfiguredTextModel: vi.fn(() => ({
    apiKey: 'test-key',
    baseUrl: 'https://example.test',
    modelId: 'test-text-model',
    protocol: 'openai-compatible',
  })),
}));
vi.mock('@/lib/nova-text-protocol', () => ({ supportsAgentNativeWebSearch: vi.fn(() => false) }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(next => {
    resolve = next;
  });
  return { promise, resolve };
}

function StrictModeWrapper({ children }: PropsWithChildren) {
  return <StrictMode>{children}</StrictMode>;
}

function proposal(overrides: Record<string, unknown> = {}) {
  return {
    action: 'generate' as const,
    prompt: '生成测试图片',
    referencedImageIds: [],
    reason: '测试提案',
    ...overrides,
  };
}

function resolvedLayout() {
  return {
    outputSize: '1K' as const,
    aspectRatio: '1:1' as const,
    temperature: 1,
    gptImageQuality: 'auto' as const,
    gptImageStyle: 'natural' as const,
    gptImageBackground: 'auto' as const,
    parallelCount: 1 as const,
  };
}

describe('useAgentChat session binding', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    store.loadAgentSession.mockResolvedValue({
      messages: [{ id: 'message-1', role: 'assistant', text: 'existing', createdAt: 1 }],
      images: [{
        imgId: 'image-1',
        source: 'uploaded',
        thumbnail: 'data:image/png;base64,AA==',
        description: 'old description',
        mimeType: 'image/png',
        createdAt: 1,
      }],
      imageModel: null,
    });
    store.loadPendingProposal.mockResolvedValue(null);
    store.loadPendingGeneration.mockResolvedValue(null);
    store.putMessage.mockResolvedValue(undefined);
    store.putImageRecord.mockResolvedValue(undefined);
    store.saveImageModel.mockResolvedValue(undefined);
    store.savePendingGeneration.mockResolvedValue(undefined);
    store.clearPendingGeneration.mockResolvedValue(undefined);
    client.describeImage.mockResolvedValue('updated description');
    client.streamAgentChat.mockReturnValue({ abort: vi.fn() });
    taskClient.createNovaTask.mockResolvedValue('task-default');
    taskClient.getNovaTask.mockResolvedValue({ status: 'processing' });
    taskClient.resolveImageTaskProvider.mockReturnValue({
      apiKey: 'image-key',
      baseUrl: 'https://example.test',
      protocol: 'openai-compatible',
      modelId: 'image-model',
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('passes the explicit sessionId to session loading and message/image persistence', async () => {
    const { result } = renderHook(() => useAgentChat('session-42'));

    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(store.loadAgentSession).toHaveBeenCalledWith('session-42');

    await act(async () => {
      result.current.clearContext();
    });
    expect(store.putMessage).toHaveBeenCalledWith(
      expect.objectContaining({ role: 'context-divider' }),
      'session-42',
    );

    await act(async () => {
      await result.current.redescribeImage('image-1');
    });
    expect(store.putImageRecord).toHaveBeenCalledWith(
      expect.objectContaining({ imgId: 'image-1', description: 'updated description' }),
      'session-42',
    );
  });

  it('does not persist a description that finishes after unmount', async () => {
    const description = deferred<string>();
    client.describeImage.mockReturnValueOnce(description.promise);
    const hook = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(hook.result.current.ready).toBe(true));

    const redescribe = hook.result.current.redescribeImage('image-1');
    hook.unmount();
    await act(async () => description.resolve('late description'));
    await redescribe;

    expect(store.putImageRecord).not.toHaveBeenCalled();
  });

  it('passes the sessionId when clearing the session', async () => {
    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));

    await act(async () => {
      await result.current.clearSession();
    });

    expect(store.clearAgentSession).toHaveBeenCalledWith('session-42');
  });

  it('passes the sessionId when retrying and deleting the previous turn', async () => {
    store.loadAgentSession.mockResolvedValueOnce({
      messages: [
        { id: 'user-1', role: 'user', text: '请求', createdAt: 1 },
        { id: 'assistant-1', role: 'assistant', text: '回复', createdAt: 2 },
      ],
      images: [],
      imageModel: null,
    });
    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));

    act(() => {
      result.current.retryMessage('user-1');
    });

    expect(store.deleteMessages).toHaveBeenCalledWith(['assistant-1'], 'session-42');
  });

  it('restores the mounted state after StrictMode effect replay', async () => {
    client.streamAgentChat.mockReturnValue({ abort: vi.fn() });
    const { result } = renderHook(() => useAgentChat('session-42'), { wrapper: StrictModeWrapper });
    await waitFor(() => expect(result.current.ready).toBe(true));

    await act(async () => {
      await result.current.sendMessage('strict-mode ping', [], []);
    });

    expect(store.putMessage).toHaveBeenCalledWith(
      expect.objectContaining({ role: 'user', text: 'strict-mode ping' }),
      'session-42',
    );
  });

  it('does not send while session restoration is still pending', async () => {
    const session = deferred<{
      messages: [{ id: string; role: 'assistant'; text: string; createdAt: number }];
      images: never[];
      imageModel: null;
    }>();
    store.loadAgentSession.mockReturnValueOnce(session.promise);

    const { result } = renderHook(() => useAgentChat('session-42'));
    expect(result.current.ready).toBe(false);

    await act(async () => {
      await result.current.sendMessage('抢跑消息', [], []);
    });

    expect(client.streamAgentChat).not.toHaveBeenCalled();
    expect(result.current.messages).toEqual([]);

    await act(async () => {
      session.resolve({
        messages: [{ id: 'restored-message', role: 'assistant' as const, text: '已恢复', createdAt: 1 }],
        images: [],
        imageModel: null,
      });
    });
    await waitFor(() => expect(result.current.ready).toBe(true));
  });

  it('disables Workspace send and upload controls before session restoration completes', async () => {
    const session = deferred<{
      messages: [];
      images: never[];
      imageModel: null;
    }>();
    store.loadAgentSession.mockReturnValueOnce(session.promise);

    const workspace = render(<AgentChatWorkspace activeSessionId="session-42" />);

    expect(screen.getByRole('textbox')).toHaveAttribute('contenteditable', 'false');
    expect(screen.getByTitle('发送')).toBeDisabled();
    expect(screen.getByTitle('上传图片')).toBeDisabled();

    await act(async () => {
      session.resolve({ messages: [], images: [], imageModel: null });
    });
    workspace.unmount();
  });

  it('persists background-approved generation data for refresh recovery', async () => {
    let onDone!: (fullText: string, parsedProposal: ReturnType<typeof proposal>, parsedProposals?: ReturnType<typeof proposal>[]) => void;
    client.streamAgentChat.mockImplementation((_request, callbacks) => {
      onDone = callbacks.onDone;
      return { abort: vi.fn() };
    });
    taskClient.createNovaTask.mockResolvedValueOnce('background-task-1');

    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));
    await act(async () => {
      await result.current.sendMessage('批量请求', [], []);
    });

    const first = proposal({ productKey: 'product-a', productName: '商品 A' });
    const next = proposal({ productKey: 'product-b', productName: '商品 B' });
    act(() => onDone('', first, [first, next]));

    await act(async () => {
      await result.current.approveProposal('确认商品 A', [], 'image-model', resolvedLayout());
    });

    await waitFor(() => expect(store.savePendingGeneration).toHaveBeenCalledWith(
      expect.objectContaining({
        taskId: 'background-task-1',
        proposal: expect.objectContaining({ productName: '商品 A' }),
      }),
      'session-42',
    ));
    act(() => result.current.stopStreaming());
  });

  it('does not resurrect a task whose creation finishes after stopStreaming', async () => {
    let onDone!: (fullText: string, parsedProposal: ReturnType<typeof proposal>, parsedProposals?: ReturnType<typeof proposal>[]) => void;
    const task = deferred<string>();
    client.streamAgentChat.mockImplementation((_request, callbacks) => {
      onDone = callbacks.onDone;
      return { abort: vi.fn() };
    });
    taskClient.createNovaTask.mockReturnValueOnce(task.promise);

    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));
    await act(async () => {
      await result.current.sendMessage('停止竞态', [], []);
    });
    const pending = proposal({ productKey: 'pending-task' });
    act(() => onDone('', pending, [pending]));

    const approval = result.current.approveProposal(pending.prompt, [], 'image-model', resolvedLayout());
    act(() => result.current.stopStreaming());
    await act(async () => {
      task.resolve('late-task');
      await approval;
    });

    expect(store.savePendingGeneration).not.toHaveBeenCalled();
    expect(taskClient.getNovaTask).not.toHaveBeenCalled();
  });

  it('keeps cancellation isolated when a later task starts', async () => {
    let onDone!: (fullText: string, parsedProposal: ReturnType<typeof proposal>, parsedProposals?: ReturnType<typeof proposal>[]) => void;
    const pollCalls: string[] = [];
    client.streamAgentChat.mockImplementation((_request, callbacks) => {
      onDone = callbacks.onDone;
      return { abort: vi.fn() };
    });
    taskClient.createNovaTask
      .mockResolvedValueOnce('task-1')
      .mockResolvedValueOnce('task-2')
      .mockResolvedValueOnce('task-3');
    taskClient.getNovaTask.mockImplementation(async (taskId: string) => {
      pollCalls.push(taskId);
      return { status: 'processing' };
    });

    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));
    vi.useFakeTimers();

    await act(async () => {
      await result.current.sendMessage('三个商品', [], []);
    });
    const first = proposal({ productKey: 'product-a' });
    const second = proposal({ productKey: 'product-b' });
    const third = proposal({ productKey: 'product-c' });
    act(() => onDone('', first, [first, second, third]));

    await act(async () => {
      await result.current.approveProposal(first.prompt, [], 'image-model', resolvedLayout());
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(pollCalls).toContain('task-1');

    await act(async () => {
      await result.current.approveProposal(second.prompt, [], 'image-model', resolvedLayout());
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(pollCalls).toContain('task-2');

    const task1CallsBeforeStop = pollCalls.filter(taskId => taskId === 'task-1').length;
    const task2CallsBeforeStop = pollCalls.filter(taskId => taskId === 'task-2').length;
    act(() => result.current.stopStreaming());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      await result.current.sendMessage('停止后新任务', [], []);
    });
    const replacement = proposal({ productKey: 'replacement' });
    act(() => onDone('', replacement, [replacement]));
    const replacementApproval = result.current.approveProposal(replacement.prompt, [], 'image-model', resolvedLayout());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      vi.advanceTimersByTime(4000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(pollCalls.filter(taskId => taskId === 'task-1').length).toBe(task1CallsBeforeStop);
    expect(pollCalls.filter(taskId => taskId === 'task-2').length).toBe(task2CallsBeforeStop);

    act(() => result.current.stopStreaming());
    await act(async () => {
      await replacementApproval;
    });
  });

  it('persists a system note when the agent stream errors', async () => {
    let onError!: (error: Error) => void;
    client.streamAgentChat.mockImplementation((_request, callbacks) => {
      onError = callbacks.onError;
      return { abort: vi.fn() };
    });

    const { result } = renderHook(() => useAgentChat('session-42'));
    await waitFor(() => expect(result.current.ready).toBe(true));
    await act(async () => {
      await result.current.sendMessage('一只红色陶瓷杯', [], []);
    });

    act(() => onError(new Error('HTTP 429: All accounts exhausted')));

    expect(result.current.error).toBe('HTTP 429: All accounts exhausted');
    expect(result.current.phase).toBe('idle');
    expect(result.current.messages.some(message => (
      message.role === 'system-note' && message.text.includes('HTTP 429: All accounts exhausted')
    ))).toBe(true);
    expect(store.putMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        role: 'system-note',
        text: '请求失败：HTTP 429: All accounts exhausted',
      }),
      'session-42',
    );
  });
});
