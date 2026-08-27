import { parseUpstreamModelList } from '@/lib/provider-registry';

export async function fetchUpstreamModels(input: {
  baseUrl: string;
  apiKey: string;
  protocol: string;
}): Promise<string[]> {
  const response = await fetch('/api/nova/proxy/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store',
    body: JSON.stringify({
      baseUrl: input.baseUrl,
      protocol: input.protocol,
      apiKey: input.apiKey,
    }),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(detail.slice(0, 180) || `读取模型列表失败（${response.status}）`);
  }
  return parseUpstreamModelList(await response.json());
}
