import * as THREE from "./module/three.module.js";
import Stats from "./module/stats.module.js";
import * as OP from "./utils/operator.js";
import * as Pedestal from "./pedestal.main.js";
import * as Floor from "./floor.main.js";

let polygon;

const container = document.getElementById("app");

let innerWidth, innerHeight;
let scene, directLight, ambientLight, renderer, raycaster, camera;
let camera_position, initial_camera_position;
let current_mouse_position, initial_mouse_position, is_mouse_down;
let stats;
let tickIntval;
let idRequestAnimationFrame;
let vertical_freedom;
let previousLookAt = new THREE.Vector3();
let lookAt = new THREE.Vector3();
let cameraCenter = new THREE.Vector3();

window.oncontextmenu = (e) => {
  e.preventDefault();
};

window.onmousedown = (e) => {
  if (e.which == 2) e.preventDefault();
};

let orientation = [45, 30];
let tour = -1;
let currentRoom = -1;
let selected_door = -1;

window.document.getElementById("btn-top-view").onclick = () => {
  orientation = [90, 35];
  vertical_freedom = 0;
  tour = -1;
  lookAt = new THREE.Vector3();
  cameraCenter = new THREE.Vector3();
};
window.document.getElementById("btn-side-view").onclick = () => {
  orientation = [45, 30];
  vertical_freedom = 0.7;
  tour = -1;
  lookAt = new THREE.Vector3();
  cameraCenter = new THREE.Vector3();
};

const createButton = (floor_cnt) => {
  for (let i = 1; i <= floor_cnt; i++) {
    const btn = window.document.createElement("button");
    btn.innerHTML = "Floor " + i;
    btn.id = "btn-floor-" + i;
    btn.className = "btn btn-warning m-2";
    btn.onclick = () => {
      tour = i - 1;
      orientation = [20, 1];
      vertical_freedom = 0;
      currentRoom = 0;
    };
    window.document.getElementById("btn-holder").appendChild(btn);
  }
};

let floors = [];
const initialization = (poly_room) => {
  polygon = poly_room;

  innerWidth = container.clientWidth;
  innerHeight = container.clientHeight;

  scene = new THREE.Scene();
  directLight = new THREE.PointLight(0xffffff);
  ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  raycaster = new THREE.Raycaster();
  camera = new THREE.PerspectiveCamera(90, innerWidth / innerHeight, 0.1, 1000);

  camera_position = {
    radius: 2,
    h_angle: 0,
    v_angle: 20,
  };
  vertical_freedom = 0.7;

  is_mouse_down = [false, false, false];

  let tmp = container.offsetTop;
  window.scrollTo(0, tmp);

  Pedestal.initialize();
  let total_width = 0;
  let base = 0;

  for (const poly of polygon) {
    if (Object.keys(poly).length == 0) continue;

    let floor = Floor.createFloor(poly);
    total_width += floor.getSize().width;
    floors.push(floor);
  }

  let gap = 10;
  total_width += (floors.length - 1) * gap; //offset
  base = (-total_width + floors[0].getSize().width) / 2;
  for (let i = 0; i < floors.length; i++) {
    floors[i].addOffset(base, 0);
    base += gap + floors[i].getSize().width;
  }

  createButton(floors.length);

  // let geom = new THREE.BoxGeometry(10, 10, 10);
  // let mesh = new THREE.MeshPhongMaterial({
  //   color: "rgb(127, 10, 50)",
  //   // wireframe: true,
  // });
  // let object = new THREE.Mesh(geom, mesh);

  // object.position.set(0, 0, 0);
  // object.castShadow = true;
  // object.receiveShadow = true;

  directLight.position.set(40, 50, 30);
  directLight.lookAt(0, 0, 0);
  directLight.castShadow = true;
  directLight.shadow.camera.near = 0.1;
  directLight.shadow.camera.far = 500;
  directLight.shadow.bias = -0.0003;
  directLight.shadow.radius = 0.5;
  directLight.shadow.normalBias = 0.01;
  directLight.shadow.mapSize.width = 1024 * 10;
  directLight.shadow.mapSize.heigh = 1024 * 10;

  scene.add(directLight);
  scene.add(ambientLight);
  for (const floor of floors) {
    for (const obj of floor.objects) {
      scene.add(obj);
    }
  }
  scene.add(Pedestal.object);
  scene.add(camera);
  scene.fog = new THREE.Fog(0xffffff, 10, 100);
  scene.background = new THREE.Color(0xffffff);

  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMapType = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;

  stats = new Stats();

  let dom = renderer.domElement;
  dom.oncontextmenu = (e) => {
    e.preventDefault();
  };

  dom.onmousemove = mouseMove;
  dom.onmousedown = mouseDown;
  dom.onmouseup = mouseUp;
  dom.onmouseout = mouseUp;
  dom.id = "3d-canvas";

  const elem = document.getElementById("3d-canvas");
  if (elem !== null) container.removeChild(elem);

  container.appendChild(dom);
  container.appendChild(stats.dom);

  stats.dom.style.position = "absolute";

  //stop all request before start new request
  cancelAnimationFrame(idRequestAnimationFrame);
  if (tickIntval) tickIntval.clear();

  animate();
  tickIntval = OP.setAsyncInterval(tick, 8);
};

window.initialization = initialization;

function animate() {
  idRequestAnimationFrame = requestAnimationFrame(animate);
  renderer.render(scene, camera);
  stats.update();
}

function tick() {
  // playSimulation = document.getElementById("is-play").checked;

  camera.updateMatrixWorld();

  mouseHandler();

  if (tour > -1) {
    lookAt = floors[tour].room_centers[currentRoom];
  }

  // if (playSimulation){
  //     intersectCheck();

  //     Ball.tick();
  //     Board.tick();
  // }
}

// function intersectCheck(){
//   Ball.collision(Board);
// }

function mouseHandler() {
  vCamPositioning();
  setCamPosition();

  if (current_mouse_position == null) return;
  (() => {
    let t = 0;
    const a = 25;
    if (is_mouse_down.some((b) => b))
      t =
        (((-a * a) /
          (Math.abs(initial_mouse_position.y - current_mouse_position.y) *
            vertical_freedom +
            a) +
          a) *
          Math.PI) /
        360;

    if (is_mouse_down[0]) {
      if (initial_mouse_position.y < current_mouse_position.y) t *= -1;
    } else if (is_mouse_down[1]) {
      t = 20 * t;
      if (initial_mouse_position.y >= current_mouse_position.y)
        camera_position.radius = initial_camera_position.radius - t;
      else camera_position.radius = initial_camera_position.radius + t;
    } else if (is_mouse_down[2]) {
      //horizontalMoving
      camera_position.h_angle =
        initial_camera_position.h_angle +
        ((initial_mouse_position.x - current_mouse_position.x) * Math.PI) /
          180 /
          5;
      //verticalMoving
      if (initial_mouse_position.y >= current_mouse_position.y)
        camera_position.v_angle = initial_camera_position.v_angle - t;
      else camera_position.v_angle = initial_camera_position.v_angle + t;
    }
  })();
}

function getMousePos(event) {
  const canvas = document.getElementById("app").children[0];
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x: x,
    y: y,
  };
}

function mouseDown(event) {
  if (event.which <= 3 && event.which > 0) {
    let i = event.which - 1;
    if (is_mouse_down.some((b) => b)) return;
    if (!is_mouse_down[i]) {
      if (i == 0) {
        is_mouse_down[i] = true;
        initial_mouse_position = getMousePos(event);
        initial_camera_position = JSON.parse(JSON.stringify(camera_position));

        if (selected_door != -1) {
          for (const door of floors[tour].door_objects) {
            if (door.object.uuid == selected_door.uuid) {
              const conn = door.connection;
              let next = -1;
              if (conn[0] == currentRoom) {
                next = conn[1];
              } else {
                next = conn[0];
              }

              currentRoom = next;
            }
          }
        }
      } else if (i > 0) {
        is_mouse_down[i] = true;
        initial_mouse_position = getMousePos(event);
        initial_camera_position = JSON.parse(JSON.stringify(camera_position));
      }
    }
  }
}

function mouseMove(event) {
  current_mouse_position = getMousePos(event);

  if (tour > -1) {
    raycaster.setFromCamera(
      new THREE.Vector2(
        (current_mouse_position.x / container.clientWidth) * 2 - 1,
        -(current_mouse_position.y / container.clientHeight) * 2 + 1
      ),
      camera
    );
    const intersect = raycaster.intersectObjects(
      floors[tour].getDoors(currentRoom)
    );
    if (intersect.length > 0) {
      const obj = intersect[0].object;
      obj.material.opacity = 0.8;
      selected_door = obj;
    } else {
      if (selected_door != -1) {
        selected_door.material.opacity = 0.2;
      }
      selected_door = -1;
    }
  }
}

function mouseUp(event) {
  if (event.which <= 3 && event.which > 0) {
    let i = event.which - 1;
    is_mouse_down[i] = false;
    initial_camera_position = JSON.parse(JSON.stringify(camera_position));
  }
}

let firstCam = true;
function setCamPosition() {
  while (camera_position.v_angle < -2 * Math.PI)
    camera_position.v_angle += Math.PI * 2;
  while (camera_position.v_angle > 2 * Math.PI)
    camera_position.v_angle -= Math.PI * 2;
  while (camera_position.h_angle < -2 * Math.PI)
    camera_position.h_angle += Math.PI * 2;
  while (camera_position.h_angle > 2 * Math.PI)
    camera_position.h_angle -= Math.PI * 2;
  camera.position.set(
    camera_position.radius *
      Math.cos(camera_position.v_angle) *
      Math.sin(camera_position.h_angle),
    camera_position.radius * Math.sin(camera_position.v_angle),
    camera_position.radius *
      Math.cos(camera_position.v_angle) *
      Math.cos(camera_position.h_angle)
  );
  camera.position.add(cameraCenter);

  // let a = new THREE.Vector3(cameraPosition.radius * Math.cos(-cameraPosition.vAngle) * Math.sin(cameraPosition.hAngle),
  // cameraPosition.radius * Math.sin(-cameraPosition.vAngle),
  // cameraPosition.radius * Math.cos(-cameraPosition.vAngle) * Math.cos(cameraPosition.hAngle)).add(camera.position);
  // camera.lookAt(a.x, a.y, a.z);

  let currentLookAt = new THREE.Vector3().copy(previousLookAt);

  if (currentLookAt.distanceTo(lookAt) > 1e-8) {
    let nowLook = new THREE.Vector3()
      .addVectors(currentLookAt.multiplyScalar(31), lookAt)
      .divideScalar(32);
    console.log(nowLook);
    camera.lookAt(nowLook);
    previousLookAt = new THREE.Vector3().copy(nowLook);
    cameraCenter = new THREE.Vector3().copy(nowLook);
  } else {
    camera.lookAt(lookAt);
    previousLookAt = new THREE.Vector3().copy(lookAt);
    cameraCenter = new THREE.Vector3().copy(lookAt);
  }
}

function vCamPositioning() {
  if (!is_mouse_down[2]) {
    const t = (orientation[0] * Math.PI) / 180;
    if (Math.abs(camera_position.v_angle - t) < 1e-8)
      camera_position.v_angle = t;
    else camera_position.v_angle = (camera_position.v_angle * 9 + t) / 10;
  }
  if (!is_mouse_down[1]) {
    const t = orientation[1];
    if (Math.abs(camera_position.radius - t) < 1e-8) camera_position.radius = t;
    else camera_position.radius = (camera_position.radius * 9 + t) / 10;
  }
}
