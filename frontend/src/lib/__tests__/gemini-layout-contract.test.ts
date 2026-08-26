import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_DEFAULTS,
  BUILTIN_IMAGE_PRESETS,
  saveRegistry,
  type ImageModelConfig,
} from '@/lib/nova-models';
import {
  getAspectRatioOptions,
  getDefaultRetryLayout,
  getValidOutputSizes,
  isRetryLayoutCompatible,
  resolveAgentLayout,
  sanitizeLayoutForModel,
  supportsAutoLayout,
} from '@/lib/model-capabilities';

const GEMINI_PRESET_ID = 'gemini-3-pro-image-preview';
const GEMINI_31_PRESET_ID = 'gemini-3.1-flash-image-preview';

const geminiConfig: ImageModelConfig = {
  ...BUILTIN_IMAGE_PRESETS[GEMINI_PRESET_ID],
  id: 'gemini-configured',
  builtinPreset: GEMINI_PRESET_ID,
  apiKey: 'test-key',
};

describe('Gemini layout contract', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('does not expose auto sizes or auto-layout support for Gemini models', () => {
    expect(supportsAutoLayout(GEMINI_PRESET_ID)).toBe(false);
    expect(getValidOutputSizes(GEMINI_PRESET_ID)).not.toContain('auto');
    expect(getDefaultRetryLayout(GEMINI_PRESET_ID)).toEqual({
      outputSize: '1K',
      aspectRatio: '1:1',
    });
    expect(isRetryLayoutCompatible(GEMINI_PRESET_ID, 'auto', 'auto')).toBe(false);

    saveRegistry({
      imageModels: [geminiConfig],
      textModels: [],
      defaults: DEFAULT_DEFAULTS,
    });
    expect(supportsAutoLayout(geminiConfig.id)).toBe(false);
    expect(getValidOutputSizes(geminiConfig.id)).not.toContain('auto');
  });

  it('never returns an auto aspect option for Gemini even if outputSize is auto', () => {
    const ratios = getAspectRatioOptions(GEMINI_PRESET_ID, 'auto');
    expect(ratios.map(option => option.value)).not.toContain('auto');
    expect(ratios.some(option => option.value === '3:4')).toBe(true);

    const flashRatios = getAspectRatioOptions(GEMINI_31_PRESET_ID, 'auto');
    expect(flashRatios.map(option => option.value)).not.toContain('auto');
    expect(flashRatios.some(option => option.value === '3:4')).toBe(true);
  });

  it('resolves agent auto intent into concrete Gemini layout values', () => {
    const layout = resolveAgentLayout(GEMINI_PRESET_ID, {
      requestedOutputSize: 'auto',
      requestedAspectRatio: '3:4',
    });
    expect(layout.outputSize).not.toBe('auto');
    expect(layout.aspectRatio).toBe('3:4');
  });

  it('clears residual auto layout when sanitizing Gemini settings', () => {
    expect(sanitizeLayoutForModel(GEMINI_PRESET_ID, 'auto', 'auto')).toEqual({
      outputSize: '1K',
      aspectRatio: '1:1',
    });
    expect(sanitizeLayoutForModel(GEMINI_PRESET_ID, '1K', 'auto')).toEqual({
      outputSize: '1K',
      aspectRatio: '1:1',
    });
    expect(sanitizeLayoutForModel(GEMINI_PRESET_ID, '1K', '3:4')).toEqual({
      outputSize: '1K',
      aspectRatio: '3:4',
    });
    expect(sanitizeLayoutForModel(GEMINI_31_PRESET_ID, 'auto', '3:4')).toEqual({
      outputSize: '1K',
      aspectRatio: '3:4',
    });
    expect(sanitizeLayoutForModel(GEMINI_PRESET_ID, '4K', 'auto')).toEqual({
      outputSize: '4K',
      aspectRatio: '1:1',
    });
    expect(sanitizeLayoutForModel(GEMINI_31_PRESET_ID, '4K', 'auto')).toEqual({
      outputSize: '4K',
      aspectRatio: '1:1',
    });
  });

  it('does not advertise 512 for Gemini 3.1 Flash but keeps official ratios', () => {
    expect(getValidOutputSizes(GEMINI_31_PRESET_ID)).toEqual(['1K', '2K', '4K']);
    expect(getValidOutputSizes(GEMINI_31_PRESET_ID)).not.toContain('512');
    expect(isRetryLayoutCompatible(GEMINI_31_PRESET_ID, '512', '1:1')).toBe(false);
    expect(isRetryLayoutCompatible(GEMINI_31_PRESET_ID, '4K', '3:4')).toBe(true);

    const flashRatios = getAspectRatioOptions(GEMINI_31_PRESET_ID, '1K').map(option => option.value);
    expect(flashRatios).toContain('3:4');
    expect(flashRatios).toEqual(expect.arrayContaining(['1:4', '1:8', '4:1', '8:1']));
  });
});
