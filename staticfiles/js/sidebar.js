document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.querySelector('[data-sidebar-backdrop]');

    if (!sidebar) return;

    const toggleSidebar = (forceOpen) => {
        const shouldOpen = typeof forceOpen === 'boolean'
            ? forceOpen
            : sidebar.classList.contains('-translate-x-full');

        sidebar.classList.toggle('-translate-x-full', !shouldOpen);

        if (backdrop) {
            backdrop.classList.toggle('hidden', !shouldOpen);
        }

        document.body.classList.toggle('overflow-hidden', shouldOpen && window.innerWidth < 1024);
    };

    const closeSidebar = () => toggleSidebar(false);
    const openSidebar = () => toggleSidebar(true);

    document.querySelectorAll('[data-sidebar-toggle]').forEach(btn => {
        btn.addEventListener('click', () => toggleSidebar());
    });

    document.querySelectorAll('[data-close-sidebar]').forEach(btn => {
        btn.addEventListener('click', closeSidebar);
    });

    if (backdrop) {
        backdrop.addEventListener('click', closeSidebar);
    }

    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeSidebar();
        }
    });

    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (window.innerWidth >= 1024) {
                openSidebar();
            } else {
                closeSidebar();
            }
        }, 150);
    });

    toggleSidebar(window.innerWidth >= 1024);
});
