(() => {
  const form = document.getElementById('chat-form');
  if (!form) return;

  const input = document.getElementById('chat-input');
  const thread = document.getElementById('chat-thread');
  const anchor = document.getElementById('chat-scroll-anchor');
  const endpoint = form.dataset.endpoint || form.action;
  const csrfToken = form.querySelector('input[name="csrfmiddlewaretoken"]')?.value;

  const escapeHtml = (str) =>
    str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');

  const scrollToBottom = () => {
    if (anchor) {
      anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  };

  const buildMessage = (role, content) => {
    const wrapper = document.createElement('div');
    const safeContent = escapeHtml(content).replace(/\n/g, '<br>');
    if (role === 'assistant') {
      wrapper.className = 'flex gap-4';
      wrapper.innerHTML = `
        <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 shadow-lg shadow-blue-500/20">
            <svg class="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
        </div>
        <div class="space-y-1">
            <div class="max-w-2xl rounded-2xl rounded-tl-none border border-white/10 bg-white/5 px-5 py-3 text-sm leading-relaxed text-zinc-100 backdrop-blur-sm">
                ${safeContent}
            </div>
        </div>
      `;
    } else {
      wrapper.className = 'flex flex-row-reverse gap-4';
      wrapper.innerHTML = `
        <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-zinc-700 text-xs text-white">
            ${form.dataset.userInitials || 'YOU'}
        </div>
        <div class="space-y-1">
            <div class="bg-gradient-to-br max-w-2xl rounded-2xl rounded-tr-none from-blue-600 to-indigo-600 px-5 py-3 text-sm leading-relaxed text-white shadow-lg shadow-blue-900/20">
                ${safeContent}
            </div>
        </div>
      `;
    }
    return wrapper;
  };

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!input.value.trim()) return;

    const messageText = input.value;
    input.value = '';
    form.querySelector('button[type="submit"]').disabled = true;

    thread.appendChild(buildMessage('user', messageText));
    scrollToBottom();

    const body = new FormData();
    body.append('message', messageText);
    if (csrfToken) {
      body.append('csrfmiddlewaretoken', csrfToken);
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body,
        headers: {
          'X-Requested-With': 'XMLHttpRequest',
          ...(csrfToken ? { 'X-CSRFToken': csrfToken } : {}),
        },
      });

      if (!response.ok) {
        throw new Error('Failed to send message.');
      }

      const data = await response.json();
      const assistantMessage = data.messages?.find((msg) => msg.role === 'assistant');
      if (assistantMessage) {
        thread.appendChild(buildMessage('assistant', assistantMessage.content));
      }
    } catch (error) {
      const errorNode = document.createElement('div');
      errorNode.className = 'rounded-2xl border border-red-500/30 bg-red-500/10 px-5 py-3 text-sm text-red-200';
      errorNode.textContent = error.message || 'Unable to reach the AI service.';
      thread.appendChild(errorNode);
    } finally {
      form.querySelector('button[type="submit"]').disabled = false;
      scrollToBottom();
    }
  });
})();

