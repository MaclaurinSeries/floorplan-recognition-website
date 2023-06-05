import * as THREE from "./module/three.module.js";

let geometry;
let mesh;
let object;

const initialize = (objectConfig) => {
  geometry = new THREE.BoxGeometry(1000, 0.05, 1000);
  mesh = new THREE.MeshStandardMaterial({
    color: 0xffffff /**wireframe : true*/,
  });
  object = new THREE.Mesh(geometry, mesh);

  object.position.set(0, 0, 0);
  object.castShadow = true;
  object.receiveShadow = true;
};

export { object, initialize };
