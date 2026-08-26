import { describe, expect, it } from 'vitest';
import { alignAgentPromptAspectRatio } from '@/lib/agent-chat-config';

describe('alignAgentPromptAspectRatio', () => {
  it('修正与比例冲突的方向词', () => {
    const aligned = alignAgentPromptAspectRatio('做一张横版商品宣传图', '9:16');
    expect(aligned).toContain('竖版');
    expect(aligned).not.toContain('横版');
    expect(aligned).toContain('9:16');
  });

  it('不破坏"横向排布"这类非构图声明的用词', () => {
    const aligned = alignAgentPromptAspectRatio('产品横向排布，竖版构图', '9:16');
    expect(aligned).toContain('横向排布');
  });

  it('不破坏"竖向滚动"这类非构图声明的用词', () => {
    const aligned = alignAgentPromptAspectRatio('页面支持竖向滚动，横版构图', '16:9');
    expect(aligned).toContain('竖向滚动');
  });
});
