import { Vector3 } from "../module/three.module.js";

const trigono = (a, b) => {
  if (a instanceof Vector3 && b instanceof Vector3) {
    let ang = a.dot(b) / (abs(a) * abs(b));
    return {
      angle: ang,
      cos: Math.cos(ang),
      sin: Math.cos(ang),
      tan: Math.tan(ang),
    };
  }
};

const abs = (a) => {
  if (a instanceof Vector3) {
    return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  }
};

const sum = (a) => {
  if (a instanceof Vector3) {
    return a.x + a.y + a.z;
  }
};

const setAsyncInterval = (func, wait, times) => {
  let interv = (function (w, t) {
    return function () {
      if (typeof t === "undefined" || t-- > 0) {
        setTimeout(interv, w);
        try {
          func.call(null);
        } catch (e) {
          t = 0;
          throw e.toString();
        }
      }
    };
  })(wait, times);
  if (interv) setTimeout(interv, wait);
  return {
    clear: () => {
      interv = null;
    },
  };
};

export { trigono, abs, sum, setAsyncInterval };
