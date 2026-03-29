/** 
 * ◎ RADIANCE PORTAL — CORE UI ENGINE (v2.3+)
 * Handles search, animations, and mobile interactions.
 */

class RadianceUI {
    constructor() {
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.injectSearchBar();
            this.setupAnimations();
            this.setupMobileDrawer();
            this.setupThemeToggle();
        });
    }

    injectSearchBar() {
        // Skip search bar on landing page for minimalist feel
        if (document.body.classList.contains('home-page')) return;

        const nav = document.querySelector('.wiki-nav');
        if (!nav || document.getElementById('searchWrapper')) return;

        const brand = nav.querySelector('.brand');
        const wrapper = document.createElement('div');
        wrapper.id = 'searchWrapper';
        wrapper.className = 'search-wrapper animate-in';
        wrapper.innerHTML = `
            <span class="search-icon">◎</span>
            <input type="text" class="search-input" placeholder="Search Radiance..." id="globalSearch">
            <div class="search-results" id="searchResults"></div>
        `;
        
        // Insert after brand
        if (brand) brand.after(wrapper);
        else nav.prepend(wrapper);

        this.initSearchLogic();
    }

    initSearchLogic() {
        const input = document.getElementById('globalSearch');
        const results = document.getElementById('searchResults');
        if (!input) return;

        input.addEventListener('input', (e) => {
            const val = e.target.value.trim().toLowerCase();
            if (val.length < 2) {
                results.classList.remove('active');
                return;
            }

            // Local Indexing (Headings, Nodes, Cards)
            const localHits = Array.from(document.querySelectorAll('h1, h2, h3, .wn-card-title, .card-title'))
                .filter(el => el.innerText.toLowerCase().includes(val))
                .slice(0, 6)
                .map(el => {
                    // Ensure element has ID for scrolling
                    if (!el.id) el.id = 'search-' + Math.random().toString(36).substr(2, 9);
                    return {
                        title: el.innerText,
                        desc: 'Documentation Resource',
                        id: el.id
                    };
                });

            if (localHits.length > 0) {
                results.innerHTML = localHits.map(h => `
                    <div class="search-item" onclick="document.getElementById('${h.id}').scrollIntoView({behavior:'smooth'}); document.getElementById('searchResults').classList.remove('active');">
                        <span class="search-item-title">${h.title}</span>
                        <span class="search-item-desc">${h.desc}</span>
                    </div>
                `).join('');
                results.classList.add('active');
            } else {
                results.classList.remove('active');
            }
        });

        // Close search on click outside
        document.addEventListener('click', (e) => {
            if (!input.contains(e.target) && !results.contains(e.target)) {
                results.classList.remove('active');
            }
        });
    }

    setupAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));
    }

    setupMobileDrawer() {
        const hamburger = document.getElementById('navHamburger');
        const drawer = document.getElementById('mobileNavDrawer');
        const overlay = document.getElementById('mobileNavOverlay');
        
        if (!hamburger || !drawer) return;

        const openMenu = () => {
            hamburger.classList.add('open');
            drawer.classList.add('open');
            document.body.style.overflow = 'hidden'; // Prevent scrolling
        };

        const closeMenu = () => {
            hamburger.classList.remove('open');
            drawer.classList.remove('open');
            document.body.style.overflow = ''; // Restore scrolling
        };

        const toggle = () => {
            const isOpen = drawer.classList.contains('open');
            isOpen ? closeMenu() : openMenu();
        };

        hamburger.addEventListener('click', (e) => {
            e.stopPropagation();
            toggle();
        });

        if (overlay) overlay.addEventListener('click', closeMenu);
        
        // Close on link click
        drawer.querySelectorAll('a').forEach(a => {
            a.addEventListener('click', () => {
                closeMenu();
            });
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && drawer.classList.contains('open')) {
                closeMenu();
            }
        });
    }

    setupThemeToggle() {
        const applyAction = (isLight) => {
            document.body.classList.toggle('light', isLight);
            const label = isLight ? '◎ Dark' : '◎ Light';
            
            // Update all theme buttons (desktop + mobile)
            document.querySelectorAll('#themeToggle, #mobileThemeToggle').forEach(btn => {
                if (btn) btn.textContent = label;
            });
            
            localStorage.setItem('radiance-theme', isLight ? 'light' : 'dark');
        };

        // Delegate click for all theme buttons
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('#themeToggle, #mobileThemeToggle');
            if (btn) {
                applyAction(!document.body.classList.contains('light'));
            }
        });
        
        // Initial state
        if (localStorage.getItem('radiance-theme') === 'light') {
            applyAction(true);
        }
    }
}

// Initializing Engine
const radianceUI = new RadianceUI();
