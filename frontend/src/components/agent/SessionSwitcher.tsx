'use client';

import { useMemo, useState, type FormEvent } from 'react';
import { Check, ChevronDown, Pencil, Plus, Trash2 } from 'lucide-react';
import { Button, buttonVariants } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { deleteAgentSessionDatabase } from '@/lib/agent-context-store';
import {
  createAgentSession,
  deleteAgentSession,
  listAgentSessions,
  renameAgentSession,
  type AgentSession,
} from '@/lib/agent-sessions';
import { cn } from '@/lib/utils';

interface SessionSwitcherProps {
  activeSessionId: string;
  onSessionChange: (sessionId: string) => void | Promise<void>;
}

type NameDialogMode = 'create' | 'rename' | null;

export function SessionSwitcher({ activeSessionId, onSessionChange }: SessionSwitcherProps) {
  const [sessions, setSessions] = useState<AgentSession[]>(() => listAgentSessions());
  const [nameDialogMode, setNameDialogMode] = useState<NameDialogMode>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [nameDraft, setNameDraft] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const activeSession = useMemo(
    () => sessions.find(session => session.id === activeSessionId)
      || { id: activeSessionId, name: '当前会话' },
    [activeSessionId, sessions],
  );

  const refreshSessions = () => setSessions(listAgentSessions());

  const handleSessionChange = async (sessionId: string) => {
    if (busy || sessionId === activeSessionId) return;

    setBusy(true);
    try {
      await onSessionChange(sessionId);
    } finally {
      setBusy(false);
    }
  };

  const openNameDialog = (mode: Exclude<NameDialogMode, null>) => {
    setNameDraft(mode === 'rename' ? activeSession.name : '');
    setNameDialogMode(mode);
  };

  const handleNameSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = nameDraft.trim();
    if (!name || !nameDialogMode || busy) return;

    setBusy(true);
    try {
      if (nameDialogMode === 'create') {
        const session = createAgentSession(name);
        refreshSessions();
        setNameDialogMode(null);
        await onSessionChange(session.id);
      } else {
        renameAgentSession(activeSessionId, name);
        refreshSessions();
        setNameDialogMode(null);
      }
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    const oldId = activeSessionId;
    if (oldId === 'default' || busy) return;

    setDeleteError(null);
    setBusy(true);
    let switchedToDefault = false;
    try {
      await onSessionChange('default');
      switchedToDefault = true;
      await deleteAgentSessionDatabase(oldId);
      deleteAgentSession(oldId);
      refreshSessions();
      setDeleteDialogOpen(false);
    } catch {
      if (switchedToDefault) {
        try { await onSessionChange(oldId); } catch { /* keep the deletion error visible */ }
      }
      setDeleteError('删除会话失败，请重试。');
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger
          className={cn(
            buttonVariants({ variant: 'outline', size: 'sm' }),
            'max-w-56 justify-between gap-2',
          )}
          aria-label="切换 Agent 会话"
          title="切换 Agent 会话"
          disabled={busy}
        >
          <span className="truncate">{activeSession.name}</span>
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-52">
          <DropdownMenuGroup>
            <DropdownMenuLabel>Agent 会话</DropdownMenuLabel>
            {sessions.map(session => (
              <DropdownMenuItem
                key={session.id}
                disabled={session.id === activeSessionId || busy}
                onClick={() => void handleSessionChange(session.id)}
              >
                <Check className={cn('size-4', session.id !== activeSessionId && 'opacity-0')} />
                <span className="min-w-0 flex-1 truncate">{session.name}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuGroup>
          <DropdownMenuSeparator />
          <DropdownMenuGroup>
            <DropdownMenuItem disabled={busy} onClick={() => openNameDialog('create')}>
              <Plus />
              新建会话
            </DropdownMenuItem>
            <DropdownMenuItem disabled={busy} onClick={() => openNameDialog('rename')}>
              <Pencil />
              重命名会话
            </DropdownMenuItem>
            {activeSessionId !== 'default' && (
              <DropdownMenuItem
                variant="destructive"
                disabled={busy}
                onClick={() => {
                  setDeleteError(null);
                  setDeleteDialogOpen(true);
                }}
              >
                <Trash2 />
                删除会话
              </DropdownMenuItem>
            )}
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        open={nameDialogMode !== null}
        onOpenChange={open => {
          if (!open && !busy) setNameDialogMode(null);
        }}
      >
        <DialogContent className="sm:max-w-sm">
          <form onSubmit={event => void handleNameSubmit(event)}>
            <DialogHeader>
              <DialogTitle>{nameDialogMode === 'create' ? '新建会话' : '重命名会话'}</DialogTitle>
              <DialogDescription>
                会话记录分别保存在本机，仅当前会话会载入 Agent。
              </DialogDescription>
            </DialogHeader>
            <label className="mt-4 block space-y-1.5 text-sm">
              <span>会话名称</span>
              <Input
                autoFocus
                aria-label="会话名称"
                value={nameDraft}
                onChange={event => setNameDraft(event.target.value)}
                maxLength={80}
              />
            </label>
            <DialogFooter className="mt-4">
              <Button type="button" variant="outline" onClick={() => setNameDialogMode(null)} disabled={busy}>
                取消
              </Button>
              <Button type="submit" disabled={!nameDraft.trim() || busy}>
                {nameDialogMode === 'create' ? '创建' : '保存'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteDialogOpen}
        onOpenChange={open => {
          if (!busy) setDeleteDialogOpen(open);
        }}
      >
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>删除会话</DialogTitle>
            <DialogDescription>
              确定删除{activeSession.name}吗？该会话的聊天记录和图片将从本机清除，且无法恢复。
            </DialogDescription>
            {deleteError && (
              <p role="alert" className="text-sm text-destructive">
                {deleteError}
              </p>
            )}
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)} disabled={busy}>
              取消
            </Button>
            <Button variant="destructive" onClick={() => void handleDelete()} disabled={busy}>
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
