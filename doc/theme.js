// xetal doc: the light/dark switch. The page follows the system's
// preference until the reader picks one; the choice is remembered.
(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("xetal-doc-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) {}
  window.addEventListener("DOMContentLoaded", function () {
    var button = document.querySelector("button.theme");
    if (!button) return;
    button.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") &&
          window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("xetal-doc-theme", next); } catch (e) {}
    });
  });
})();
