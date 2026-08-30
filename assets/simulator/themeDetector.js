(function () {
  const LIGHT_THEME_ID = "shoelace-theme-light";
  const DARK_THEME_ID = "shoelace-theme-dark";

  const matchDarkQuery = window.matchMedia?.("(prefers-color-scheme: dark)");

  const isDarkScheme = (scheme) => {
    if (!scheme) {
      return matchDarkQuery?.matches ?? false;
    }

    const normalized = scheme.toLowerCase();
    return (
      normalized.includes("dark") ||
      normalized.includes("slate") ||
      normalized.includes("night")
    );
  };

  const getActiveScheme = () => {
    const candidates = [
      document.body?.getAttribute("data-md-color-scheme"),
      document.documentElement?.getAttribute("data-md-color-scheme"),
    ];
    return (
      candidates.find(
        (value) => typeof value === "string" && value.length > 0
      ) || null
    );
  };

  const applyShoelaceTheme = (darkMode) => {
    const lightLink = document.getElementById(LIGHT_THEME_ID);
    const darkLink = document.getElementById(DARK_THEME_ID);
    const simulatorRoot = document.querySelector(".aod-simulator");

    if (lightLink) {
      lightLink.disabled = !!darkMode;
    }

    if (darkLink) {
      darkLink.disabled = !darkMode;
    }

    if (simulatorRoot) {
      simulatorRoot.classList.toggle("sl-theme-dark", !!darkMode);
      simulatorRoot.classList.toggle("sl-theme-light", !darkMode);
    }
  };

  const syncTheme = () => {
    applyShoelaceTheme(isDarkScheme(getActiveScheme()));
  };

  const initThemeSync = () => {
    syncTheme();

    if (document.body) {
      const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
          if (
            mutation.type === "attributes" &&
            mutation.attributeName === "data-md-color-scheme"
          ) {
            syncTheme();
            break;
          }
        }
      });

      observer.observe(document.body, {
        attributes: true,
        attributeFilter: ["data-md-color-scheme"],
      });
    }

    matchDarkQuery?.addEventListener("change", syncTheme);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initThemeSync, {
      once: true,
    });
  } else {
    initThemeSync();
  }
})();
