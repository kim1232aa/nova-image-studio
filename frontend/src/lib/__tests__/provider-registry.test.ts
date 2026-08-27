import { afterEach, describe, expect, it } from 'vitest';
import {
  addManualProviderModel,
  guessModelUses,
  mergeFetchedModels,
  migrateLegacyProviders,
  parseUpstreamModelList,
  toggleProviderModelUse,
} from '@/lib/provider-registry';
import {
  deriveImageAndTextModels,
  loadRegistry,
  saveRegistry,
  type ImageModelConfig,
  type TextModelConfig,
} from '@/lib/nova-models';

describe('provider registry', () => {
  afterEach(() => {
    localStorage.clear();
  });

  it('parses OpenAI and Gemini /models payloads', () => {
    expect(parseUpstreamModelList({
      data: [{ id: 'gpt-4o-mini' }, { model: 'grok-imagine-image' }, { id: 'gpt-4o-mini' }],
    })).toEqual(['gpt-4o-mini', 'grok-imagine-image']);

    expect(parseUpstreamModelList({
      models: [{ name: 'models/gemini-2.5-flash' }, 'claude-sonnet'],
    })).toEqual(['gemini-2.5-flash', 'claude-sonnet']);
  });

  it('guesses uses from model ids without treating every chat model as image', () => {
    expect(guessModelUses('gpt-4o-mini')).toEqual(['text']);
    expect(guessModelUses('gemini-3-pro-image-preview')).toEqual(['image']);
    expect(guessModelUses('grok-imagine-image')).toEqual(['image']);
    expect(guessModelUses('grok-imagine-video')).toEqual(['video']);
    expect(guessModelUses('sora-2')).toEqual(['video']);
    expect(guessModelUses('whisper-1')).toEqual(['audio']);
  });

  it('parses OpenAI data arrays that contain bare strings', () => {
    expect(parseUpstreamModelList({ data: ['gpt-4o-mini', { id: 'grok-4.3' }] })).toEqual([
      'gpt-4o-mini',
      'grok-4.3',
    ]);
  });

  it('keeps existing uses when merging fetched ids and only guesses for new rows', () => {
    const merged = mergeFetchedModels(
      [{ modelId: 'gpt-4o-mini', name: 'Mini', uses: ['text', 'image'] }],
      ['gpt-4o-mini', 'grok-imagine-image'],
    );
    expect(merged).toEqual([
      { modelId: 'gpt-4o-mini', name: 'Mini', uses: ['text', 'image'] },
      { modelId: 'grok-imagine-image', name: 'grok-imagine-image', uses: ['image'] },
    ]);
  });

  it('migrates old text/image rows that share a key into one provider', () => {
    const imageModels: ImageModelConfig[] = [{
      id: 'img_old',
      protocol: 'openai',
      name: 'Banana',
      modelId: 'gemini-3-pro-image-preview',
      apiKey: 'test-key',
      baseUrl: 'https://example.test/v1',
      builtinPreset: 'antigravity-gemini-image',
      maxRefImages: 14,
      maxOutputSize: '4K',
      supportsAdvancedParams: false,
    }];
    const textModels: TextModelConfig[] = [{
      id: 'txt_old',
      protocol: 'openai-chat-completions',
      name: 'Chat',
      modelId: 'gpt-4o-mini',
      apiKey: 'test-key',
      baseUrl: 'https://example.test/v1',
    }];

    const providers = migrateLegacyProviders(imageModels, textModels);
    expect(providers).toHaveLength(1);
    expect(providers[0].apiKey).toBe('test-key');
    expect(providers[0].baseUrl).toBe('https://example.test/v1');
    expect(providers[0].models.map((entry) => entry.modelId).sort()).toEqual([
      'gemini-3-pro-image-preview',
      'gpt-4o-mini',
    ]);

    const derived = deriveImageAndTextModels(providers);
    expect(derived.imageModels[0].id).toBe('img_old');
    expect(derived.textModels[0].id).toBe('txt_old');
    expect(derived.imageModels[0].protocol).toBe('openai');
    expect(derived.textModels[0].protocol).toBe('openai-chat-completions');
  });

  it('loads legacy localStorage as providers and keeps derived lists', () => {
    localStorage.setItem('nova-model-registry', JSON.stringify({
      imageModels: [{
        id: 'img_old',
        protocol: 'openai',
        name: 'GPT Image',
        modelId: 'gpt-image-2',
        apiKey: 'test-key',
        baseUrl: 'https://example.test',
        builtinPreset: 'gpt-image-2',
        maxRefImages: 4,
        maxOutputSize: '1K',
        supportsAdvancedParams: true,
      }],
      textModels: [{
        id: 'txt_old',
        protocol: 'openai-chat-completions',
        name: 'Chat',
        modelId: 'gpt-4o-mini',
        apiKey: 'test-key',
        baseUrl: 'https://example.test',
      }],
      defaults: {},
    }));

    const registry = loadRegistry();
    expect(registry.providers).toHaveLength(1);
    expect(registry.imageModels[0].id).toBe('img_old');
    expect(registry.textModels[0].id).toBe('txt_old');
  });

  it('openai-compatible gemini image models keep the openai protocol', () => {
    const derived = deriveImageAndTextModels([{
      id: 'prov_1',
      name: 'Gateway',
      kind: 'openai-compatible',
      apiKey: 'test-key',
      baseUrl: 'https://example.test',
      models: [{
        modelId: 'gemini-3-pro-image-preview',
        name: 'Gemini Image',
        uses: ['image'],
      }],
    }]);
    expect(derived.imageModels[0].protocol).toBe('openai');
    expect(derived.imageModels[0].builtinPreset).toBe('antigravity-gemini-image');
  });

  it('toggles uses and adds manual model ids', () => {
    const provider = addManualProviderModel({
      id: 'prov_1',
      name: 'Gateway',
      kind: 'openai-compatible',
      apiKey: 'test-key',
      baseUrl: 'https://example.test',
      models: [],
    }, 'my-hidden-model');
    expect(provider.models[0]).toMatchObject({ modelId: 'my-hidden-model', manual: true, uses: [] });
    const toggled = toggleProviderModelUse(provider, 'my-hidden-model', 'text');
    expect(toggled.models[0].uses).toEqual(['text']);
  });

  it('does not persist private gateway hostnames from tests', () => {
    saveRegistry({
      providers: [{
        id: 'prov_1',
        name: 'Example',
        kind: 'openai-compatible',
        apiKey: 'test-key',
        baseUrl: 'https://example.test',
        models: [{ modelId: 'gpt-4o-mini', name: 'Mini', uses: ['text'] }],
      }],
      imageModels: [],
      textModels: [],
      defaults: {
        textToImage: '',
        imageToImage: '',
        reversePrompt: '',
        agent: '',
        promptOptimize: '',
        imageDescribe: '',
        sliceDecomposition: '',
        sliceReconstruct: '',
        sliceImageEdit: '',
      },
    });
    expect(localStorage.getItem('nova-model-registry') || '').not.toMatch(/alibb123|ccwu\.cc/);
  });
});
