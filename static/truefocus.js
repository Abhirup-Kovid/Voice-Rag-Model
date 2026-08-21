/* TrueFocus (React Bits) reimplemented in vanilla JS — auto mode.
   Sentence words blur; a glowing corner-frame sweeps across each word in turn. */
(function () {
    "use strict";

    var container = document.getElementById("heroTitle");
    if (!container) return;
    var words = Array.prototype.slice.call(container.querySelectorAll(".focus-word"));
    if (words.length === 0) return;
    var frame = document.getElementById("focusFrame");
    if (!frame) return;

    var blurAmount = 5;
    var animationDuration = 0.5;   // seconds per word sweep
    var pauseBetweenAnimations = 1; // seconds between words
    var currentIndex = 0;

    function applyBlur() {
        words.forEach(function (w, i) {
            w.style.filter = i === currentIndex ? "blur(0px)" : "blur(" + blurAmount + "px)";
        });
    }

    function placeFrame() {
        if (!words[currentIndex]) return;
        var parentRect = container.getBoundingClientRect();
        var activeRect = words[currentIndex].getBoundingClientRect();
        frame.style.transform =
            "translate(" + (activeRect.left - parentRect.left) + "px," +
            (activeRect.top - parentRect.top) + "px)";
        frame.style.width = activeRect.width + "px";
        frame.style.height = activeRect.height + "px";
    }

    function advance() {
        currentIndex = (currentIndex + 1) % words.length;
        applyBlur();
        placeFrame();
    }

    applyBlur();
    placeFrame();
    setInterval(advance, (animationDuration + pauseBetweenAnimations) * 1000);

    window.addEventListener("resize", placeFrame);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(placeFrame);
    }
})();