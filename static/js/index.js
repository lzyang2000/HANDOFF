window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        if (dropdown) dropdown.classList.remove('show');
        if (button) button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel: play each clip to its end, then advance to the next slide.
// (Clip lengths differ, so we drive the carousel off the video instead of a fixed timer.)

// The slide currently centered in the carousel viewport (works despite the loop clones).
function getCenteredVideo() {
    const root = document.querySelector('.results-carousel');
    if (!root) return null;
    const rb = root.getBoundingClientRect();
    const center = rb.left + rb.width / 2;
    let best = null, bestDist = Infinity;
    root.querySelectorAll('video').forEach(v => {
        const r = v.getBoundingClientRect();
        if (r.width === 0) return; // skip hidden clones
        const d = Math.abs((r.left + r.width / 2) - center);
        if (d < bestDist) { bestDist = d; best = v; }
    });
    return best;
}

// Play the centered clip; pause the rest. restart=true rewinds it to the beginning.
function playCenteredVideo(restart) {
    const active = getCenteredVideo();
    document.querySelectorAll('.results-carousel video').forEach(v => {
        if (v !== active) { try { v.pause(); } catch (e) {} }
    });
    if (active) {
        if (restart) { try { active.currentTime = 0; } catch (e) {} }
        active.play().catch(() => {});
    }
}

$(document).ready(function() {

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: false,   // advance when each clip ends, not on a timer
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    var carousel = (carousels && carousels.length) ? carousels[0] : null;

    bulmaSlider.attach();

    // When a slide finishes transitioning, rewind+play the newly centered clip.
    if (carousel) {
        carousel.on('after:show', function () {
            setTimeout(function () { playCenteredVideo(true); }, 350);
        });
    }

    // When the centered clip ends, move to the next slide.
    document.querySelectorAll('.results-carousel video').forEach(function (v) {
        v.addEventListener('ended', function () {
            if (carousel) { carousel.next(); }
        });
    });

    // Pause when the carousel scrolls out of view; resume (without restarting) when back.
    const carouselRoot = document.querySelector('.results-carousel');
    if (carouselRoot) {
        const obs = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) {
                    playCenteredVideo(false);
                } else {
                    document.querySelectorAll('.results-carousel video').forEach(function (v) {
                        try { v.pause(); } catch (err) {}
                    });
                }
            });
        }, { threshold: 0.4 });
        obs.observe(carouselRoot);
    }

    // Kick off the first clip.
    setTimeout(function () { playCenteredVideo(true); }, 500);

})
