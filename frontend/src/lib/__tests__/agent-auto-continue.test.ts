import { describe, expect, it } from 'vitest';
import { extractProductLinks } from '@/lib/agent-chat-config';

describe('extractProductLinks', () => {
  it('从用户消息提取商品链接并归一化去重', () => {
    const links = extractProductLinks(
      'https://item.taobao.com/item.htm?id=111&mi_id=x  https://item.taobao.com/item.htm?id=222 两个链接帮我做图 https://item.taobao.com/item.htm?id=111&spm=y',
    );
    expect(links).toEqual(['taobao:111', 'taobao:222']);
  });

  it('没有链接时返回空数组', () => {
    expect(extractProductLinks('帮我画一只猫')).toEqual([]);
  });
});
