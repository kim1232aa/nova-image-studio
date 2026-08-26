import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';
import {
  streamAgentChat,
  type StreamAgentCallbacks,
  type StreamAgentInput,
} from '@/lib/agent-chat-client';
import type { TextProviderProtocol } from '@/lib/nova-text-protocol';

// ===== SSE 构造辅助 =====

function dataFrame(payload: unknown, event?: string): string {
  const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
  return `${event ? `event: ${event}\n` : ''}data: ${data}\n\n`;
}

function sseResponse(frames: string[]): Response {
  return new Response(frames.join(''), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

/** 各协议「纯文本一句话然后结束」的收尾流 */
function finalTextFrames(protocol: TextProviderProtocol, text: string): string[] {
  if (protocol === 'openai-chat-completions') {
    return [
      dataFrame({ choices: [{ delta: { content: text } }] }),
      dataFrame('[DONE]'),
    ];
  }
  if (protocol === 'anthropic-messages') {
    return [
      dataFrame({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
      dataFrame({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }),
      dataFrame({ type: 'message_stop' }),
    ];
  }
  if (protocol === 'google-gemini') {
    return [dataFrame({ candidates: [{ content: { parts: [{ text }] } }] })];
  }
  return [
    dataFrame({ type: 'response.output_text.delta', delta: text }),
    dataFrame({ type: 'response.completed', response: { output_text: text, output: [] } }),
  ];
}

/** 各协议「调用一次指定工具然后结束」的流 */
function toolCallFrames(protocol: TextProviderProtocol, name: string, args: Record<string, unknown>, id: string): string[] {
  const argsJson = JSON.stringify(args);
  if (protocol === 'openai-chat-completions') {
    return [
      dataFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id, function: { name, arguments: '' } }] } }] }),
      dataFrame({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: argsJson } }] } }] }),
      dataFrame('[DONE]'),
    ];
  }
  if (protocol === 'anthropic-messages') {
    return [
      dataFrame({ type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id, name, input: {} } }),
      dataFrame({ type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: argsJson } }),
      dataFrame({ type: 'message_stop' }),
    ];
  }
  if (protocol === 'google-gemini') {
    return [dataFrame({ candidates: [{ content: { parts: [{ functionCall: { name, args } }] } }] })];
  }
  return [
    dataFrame({ type: 'response.output_item.added', output_index: 1, item: { id: `item_${id}`, call_id: id, type: 'function_call', name, arguments: '' } }),
    dataFrame({ type: 'response.function_call_arguments.done', output_index: 1, item_id: `item_${id}`, arguments: argsJson }),
    dataFrame({ type: 'response.completed', response: { output_text: '', output: [{ type: 'function_call', id: `item_${id}`, call_id: id, name, arguments: argsJson }] } }),
  ];
}

// ===== 测试辅助 =====

function baseInput(protocol: TextProviderProtocol, extra?: Partial<StreamAgentInput>): StreamAgentInput {
  return {
    apiKey: 'test-key',
    model: 'test-model',
    protocol,
    history: [{ id: 'u1', role: 'user', text: '帮我看看这个链接', createdAt: 1 }],
    catalog: [],
    modelCatalog: [],
    ...extra,
  };
}

interface Collected {
  deltas: string[];
  toolActivity: string[];
  done: { text: string; proposal: { prompt?: string } | null; proposals?: Array<{ prompt?: string }> } | null;
  error: Error | null;
}

function makeCallbacks(): { callbacks: StreamAgentCallbacks; collected: Collected } {
  const collected: Collected = { deltas: [], toolActivity: [], done: null, error: null };
  const callbacks: StreamAgentCallbacks = {
    onDelta: token => collected.deltas.push(token),
    onReasoning: () => undefined,
    onToolActivity: text => collected.toolActivity.push(text),
    onDone: ((text, proposal, proposals) => { collected.done = { text, proposal, proposals }; }) as StreamAgentCallbacks['onDone'],
    onError: err => { collected.error = err; },
  };
  return { callbacks, collected };
}

function requestBodyOf(mock: Mock, callIndex: number): Record<string, unknown> {
  const [, init] = mock.mock.calls[callIndex] as [string, RequestInit];
  return (JSON.parse(String(init.body)) as { requestBody: Record<string, unknown> }).requestBody;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const ALL_PROTOCOLS: TextProviderProtocol[] = [
  'openai-chat-completions',
  'anthropic-messages',
  'google-gemini',
  'openai-responses',
];

describe('buildAgentRequestBody 的 CDP 工具声明', () => {
  it.each(ALL_PROTOCOLS)('%s：cdp 开启时声明浏览器工具', async protocol => {
    const mock = vi.fn().mockResolvedValue(sseResponse(finalTextFrames(protocol, '好的')));
    vi.stubGlobal('fetch', mock);
    const { callbacks } = makeCallbacks();

    await streamAgentChat(baseInput(protocol, { cdp: true, cdpExecutor: async () => 'ok' }), callbacks).promise;

    const body = requestBodyOf(mock, 0);
    const tools = (body.tools || []) as Array<Record<string, unknown>>;
    let names: string[] = [];
    if (protocol === 'openai-chat-completions') {
      names = tools.map(t => (t.function as { name: string }).name);
    } else if (protocol === 'google-gemini') {
      names = ((tools[0].function_declarations || []) as Array<{ name: string }>).map(t => t.name);
    } else {
      names = tools.map(t => t.name as string);
    }
    for (const toolName of ['browser_status', 'browser_set_port', 'browser_list_tabs', 'browser_open_url', 'browser_read_page', 'browser_read_taobao', 'browser_save_images']) {
      expect(names).toContain(toolName);
    }
  });

  it('cdp 关闭时不声明浏览器工具', async () => {
    const mock = vi.fn().mockResolvedValue(sseResponse(finalTextFrames('openai-chat-completions', '好的')));
    vi.stubGlobal('fetch', mock);
    const { callbacks } = makeCallbacks();

    await streamAgentChat(baseInput('openai-chat-completions'), callbacks).promise;

    const body = requestBodyOf(mock, 0);
    const names = ((body.tools || []) as Array<{ function: { name: string } }>).map(t => t.function.name);
    expect(names).toEqual(['propose_image_action']);
  });
});

describe('CDP 工具循环', () => {
  it.each(ALL_PROTOCOLS)('%s：工具调用执行后按协议原生格式回灌并继续', async protocol => {
    const executor = vi.fn(async () => '工具执行结果文本');
    const mock = vi.fn()
      .mockResolvedValueOnce(sseResponse(toolCallFrames(protocol, 'browser_open_url', { url: 'https://item.taobao.com/item.htm?id=1' }, 'call_1')))
      .mockResolvedValueOnce(sseResponse(finalTextFrames(protocol, '已打开页面')));
    vi.stubGlobal('fetch', mock);
    const { callbacks, collected } = makeCallbacks();

    await streamAgentChat(baseInput(protocol, { cdp: true, cdpExecutor: executor }), callbacks).promise;

    expect(collected.error).toBeNull();
    expect(executor).toHaveBeenCalledTimes(1);
    expect(executor).toHaveBeenCalledWith('browser_open_url', { url: 'https://item.taobao.com/item.htm?id=1' }, expect.any(Function));
    expect(collected.toolActivity.some(text => text.includes('browser_open_url'))).toBe(true);
    expect(mock).toHaveBeenCalledTimes(2);
    expect(collected.done?.proposal).toBeNull();
    expect(collected.done?.text).toContain('工具执行结果文本');
    expect(collected.done?.text).toContain('已打开页面');

    // 第二轮请求体必须包含协议原生的工具调用与结果回灌
    const body = requestBodyOf(mock, 1);
    if (protocol === 'openai-chat-completions') {
      const messages = body.messages as Array<Record<string, unknown>>;
      const assistantToolMsg = messages.find(m => m.role === 'assistant' && Array.isArray(m.tool_calls));
      expect(assistantToolMsg).toBeTruthy();
      const toolCalls = assistantToolMsg!.tool_calls as Array<{ id: string; function: { name: string; arguments: string } }>;
      expect(toolCalls[0].function.name).toBe('browser_open_url');
      const toolMsg = messages.find(m => m.role === 'tool');
      expect(toolMsg).toMatchObject({ tool_call_id: 'call_1', content: '工具执行结果文本' });
    } else if (protocol === 'anthropic-messages') {
      const messages = body.messages as Array<{ role: string; content: Array<Record<string, unknown>> }>;
      const assistantToolMsg = messages.find(m => m.role === 'assistant' && m.content.some(p => p.type === 'tool_use'));
      expect(assistantToolMsg).toBeTruthy();
      const userToolMsg = messages.find(m => m.role === 'user' && Array.isArray(m.content) && m.content.some(p => p.type === 'tool_result'));
      expect(userToolMsg).toBeTruthy();
      expect(userToolMsg!.content.find(p => p.type === 'tool_result')).toMatchObject({ tool_use_id: 'call_1', content: '工具执行结果文本' });
    } else if (protocol === 'google-gemini') {
      const contents = body.contents as Array<{ role: string; parts: Array<Record<string, unknown>> }>;
      const modelMsg = contents.find(c => c.role === 'model' && c.parts.some(p => p.functionCall));
      expect(modelMsg).toBeTruthy();
      const userMsg = contents.find(c => c.role === 'user' && c.parts.some(p => p.functionResponse));
      expect(userMsg).toBeTruthy();
      const fr = userMsg!.parts.find(p => p.functionResponse)!.functionResponse as { name: string; response: { result: string } };
      expect(fr.name).toBe('browser_open_url');
      expect(fr.response.result).toBe('工具执行结果文本');
    } else {
      const input = body.input as Array<Record<string, unknown>>;
      const call = input.find(item => item.type === 'function_call');
      expect(call).toMatchObject({ call_id: 'call_1', name: 'browser_open_url' });
      const output = input.find(item => item.type === 'function_call_output');
      expect(output).toMatchObject({ call_id: 'call_1', output: '工具执行结果文本' });
    }
  });

  it('模型调用 propose_image_action 时直接产出提案，不执行 CDP 工具', async () => {
    const executor = vi.fn(async () => '不应被调用');
    const proposalArgs = {
      action: 'generate',
      prompt: '画一张淘宝主图',
      referenced_image_ids: [],
      reason: '用户要做主图',
    };
    const mock = vi.fn().mockResolvedValue(
      sseResponse(toolCallFrames('openai-chat-completions', 'propose_image_action', proposalArgs, 'call_p')),
    );
    vi.stubGlobal('fetch', mock);
    const { callbacks, collected } = makeCallbacks();

    await streamAgentChat(baseInput('openai-chat-completions', { cdp: true, cdpExecutor: executor }), callbacks).promise;

    expect(executor).not.toHaveBeenCalled();
    expect(mock).toHaveBeenCalledTimes(1);
    expect(collected.done?.proposal?.prompt).toBe('画一张淘宝主图');
  });

  it('同一轮返回多个商品提案时按顺序交给调用方排队', async () => {
    const first = {
      action: 'generate',
      prompt: '商品 A 的宣传图',
      referenced_image_ids: ['img_a'],
      reason: '商品 A',
    };
    const second = {
      action: 'generate',
      prompt: '商品 B 的宣传图',
      referenced_image_ids: ['img_b'],
      reason: '商品 B',
    };
    const mock = vi.fn().mockResolvedValue(sseResponse([
      dataFrame({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_a', function: { name: 'propose_image_action', arguments: JSON.stringify(first) } }, { index: 1, id: 'call_b', function: { name: 'propose_image_action', arguments: JSON.stringify(second) } }] } }] }),
      dataFrame('[DONE]'),
    ]));
    vi.stubGlobal('fetch', mock);
    const { callbacks, collected } = makeCallbacks();

    await streamAgentChat(baseInput('openai-chat-completions'), callbacks).promise;

    expect(collected.error).toBeNull();
    expect(collected.done?.proposal?.prompt).toBe('商品 A 的宣传图');
    expect(collected.done?.proposals?.map(item => item.prompt)).toEqual(['商品 A 的宣传图', '商品 B 的宣传图']);
  });

  it('模型连续调用工具超过最大轮数时强制收尾', async () => {
    const executor = vi.fn(async () => 'ok');
    const mock = vi.fn().mockImplementation(async () =>
      sseResponse(toolCallFrames('openai-chat-completions', 'browser_list_tabs', {}, 'call_x')),
    );
    vi.stubGlobal('fetch', mock);
    const { callbacks, collected } = makeCallbacks();

    await streamAgentChat(baseInput('openai-chat-completions', { cdp: true, cdpExecutor: executor }), callbacks).promise;

    expect(mock).toHaveBeenCalledTimes(12);
    expect(executor).toHaveBeenCalledTimes(11);
    expect(collected.done?.proposal).toBeNull();
    expect(collected.done?.text).toContain('ok');
    expect(collected.toolActivity[0]).toContain('正在连接模型');
  });

  it('提案与浏览器工具同轮出现时，提案不丢失，与后续轮提案一起排队', async () => {
    const proposalA = {
      action: 'generate',
      prompt: '商品 A 的宣传图',
      referenced_image_ids: ['img_a1'],
      reason: '先做商品 A',
      product_key: 'https://item.taobao.com/item.htm?id=1',
    };
    const proposalB = {
      action: 'generate',
      prompt: '商品 B 的宣传图',
      referenced_image_ids: ['img_b1'],
      reason: '再做商品 B',
      product_key: 'https://item.taobao.com/item.htm?id=2',
    };
    // 第 1 轮：模型边给商品 A 提案、边调用浏览器打开商品 B（真实多商品场景的常见模式）
    const round1 = sseResponse([
      dataFrame({ choices: [{ delta: { tool_calls: [
        { index: 0, id: 'call_pa', function: { name: 'propose_image_action', arguments: JSON.stringify(proposalA) } },
        { index: 1, id: 'call_open', function: { name: 'browser_open_url', arguments: JSON.stringify({ url: 'https://item.taobao.com/item.htm?id=2' }) } },
      ] } }] }),
      dataFrame('[DONE]'),
    ]);
    const round2 = sseResponse(toolCallFrames('openai-chat-completions', 'propose_image_action', proposalB, 'call_pb'));
    const executor = vi.fn(async () => '已打开商品 B 页面');
    const mock = vi.fn()
      .mockResolvedValueOnce(round1)
      .mockResolvedValueOnce(round2);
    vi.stubGlobal('fetch', mock);
    const { callbacks, collected } = makeCallbacks();

    await streamAgentChat(baseInput('openai-chat-completions', { cdp: true, cdpExecutor: executor }), callbacks).promise;

    expect(executor).toHaveBeenCalledTimes(1);
    expect(collected.error).toBeNull();
    expect(collected.done?.proposal?.prompt).toBe('商品 A 的宣传图');
    expect(collected.done?.proposals?.map(item => item.prompt)).toEqual(['商品 A 的宣传图', '商品 B 的宣传图']);
  });
});
