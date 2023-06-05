import * as THREE from "./module/three.module.js";
import { BufferGeometryUtils } from "./module/bufferGeometryUtils.js";

const commonColors = [
  { name: "Soft Red", rgb: 0xff5a5f, hex: "#FF5A5F" },
  { name: "Soft Green", rgb: 0x70c1b3, hex: "#70C1B3" },
  { name: "Soft Blue", rgb: 0x6d8ea0, hex: "#6D8EA0" },
  { name: "Soft Yellow", rgb: 0xffd23f, hex: "#FFD23F" },
  { name: "Soft Orange", rgb: 0xff9a00, hex: "#FF9A00" },
  { name: "Soft Purple", rgb: 0xb83b5e, hex: "#B83B5E" },
  { name: "Soft Pink", rgb: 0xff7b6b, hex: "#FF7B6B" },
  { name: "Soft Brown", rgb: 0xdab785, hex: "#DAB785" },
  { name: "Soft Gray", rgb: 0xb2b2b2, hex: "#B2B2B2" },
  { name: "Soft Maroon", rgb: 0xb22222, hex: "#B22222" },
];

class Floor {
  constructor() {
    this.wall = [];
    this.window = [];
    this.door = [];
    this.room = [];

    this.connection = [];
    this.door_objects = [];
    this.room_centers = [];

    this.objects = [];
    this.bounding_box = {
      top_left: {
        x: Infinity,
        y: Infinity,
      },
      bottom_right: {
        x: 0,
        y: 0,
      },
    };

    this.wall_color = 0xffddab;
    this.extrude_steps = 100;
  }

  createWall() {
    const extrude_settings = {
      steps: this.extrude_steps,
      depth: this.wall_height,
      bevelEnabled: false,
    };
    const geometries = [];
    for (const poly of this.wall) {
      const shape = new THREE.Shape();
      let start = true;
      for (let [u, v] of poly) {
        u = u * this.width;
        v = v * this.height;

        this.bounding_box.top_left.x = Math.min(
          this.bounding_box.top_left.x,
          u
        );
        this.bounding_box.top_left.y = Math.min(
          this.bounding_box.top_left.y,
          v
        );
        this.bounding_box.bottom_right.x = Math.max(
          this.bounding_box.bottom_right.x,
          u
        );
        this.bounding_box.bottom_right.y = Math.max(
          this.bounding_box.bottom_right.y,
          v
        );
        if (start) {
          start = false;
          shape.moveTo(u, v);
        }
        shape.lineTo(u, v);
      }

      const geometry = new THREE.ExtrudeGeometry(shape, extrude_settings);
      geometries.push(geometry);
    }

    const geometry = BufferGeometryUtils.mergeBufferGeometries(geometries);
    const material = new THREE.MeshStandardMaterial({
      color: this.wall_color,
      // wireframe: true,
    });
    const object = new THREE.Mesh(geometry, material);
    object.position.set(-this.width / 2, 0, -this.height / 2);
    object.castShadow = true;
    object.receiveShadow = true;

    object.rotateX(Math.PI / 2);
    object.translateZ(-this.wall_height);

    this.objects.push(object);
  }

  createDoor(connection) {
    const extrude_settings = {
      steps: this.extrude_steps,
      depth: this.wall_height - this.door_height,
      bevelEnabled: false,
    };
    const geometries = [];
    const doors_ = [];
    const door_material = [];
    for (const poly of this.door) {
      const shape = new THREE.Shape();
      let start = true;
      for (let [u, v] of poly) {
        u = u * this.width;
        v = v * this.height;

        if (start) {
          start = false;
          shape.moveTo(u, v);
        }
        shape.lineTo(u, v);
      }

      const geometry = new THREE.ExtrudeGeometry(shape, extrude_settings);
      geometry.translate(0, 0, 0);
      geometries.push(geometry);

      const door_geom = new THREE.ExtrudeGeometry(shape, {
        steps: this.extrude_steps,
        depth: this.door_height,
        bevelEnabled: false,
      });
      door_geom.translate(0, 0, 0);
      const material = new THREE.MeshStandardMaterial({
        color: 0xff8000,
        transparent: true,
        opacity: 0.2,
      });
      const door_obj = new THREE.Mesh(door_geom, material);

      door_obj.position.set(-this.width / 2, 0, -this.height / 2);
      door_obj.castShadow = true;
      door_obj.receiveShadow = true;

      door_obj.rotateX(Math.PI / 2);
      door_obj.translateZ(-this.door_height);

      door_material.push(material);
      doors_.push(door_obj);
    }

    for (let i = 0; i < connection.length; i++) {
      if (connection[i] == null) continue;

      this.door_objects.push({
        object: doors_[i],
        connection: connection[i],
      });
      this.objects.push(doors_[i]);
    }

    const geometry = BufferGeometryUtils.mergeBufferGeometries(geometries);
    const material = new THREE.MeshStandardMaterial({
      color: this.wall_color,
      // wireframe: true,
    });
    const object = new THREE.Mesh(geometry, material);
    object.position.set(-this.width / 2, 0, -this.height / 2);
    object.castShadow = true;
    object.receiveShadow = true;

    object.rotateX(Math.PI / 2);
    object.translateZ(-this.wall_height);

    this.objects.push(object);
  }

  createWindow() {
    const extrude_settings = {
      steps: 2,
      depth: (this.wall_height - this.window_height) / 2,
      bevelEnabled: false,
    };
    const geometries = [];
    for (const poly of this.window) {
      const shape = new THREE.Shape();
      let start = true;
      for (let [u, v] of poly) {
        u = u * this.width;
        v = v * this.height;

        if (start) {
          start = false;
          shape.moveTo(u, v);
        }
        shape.lineTo(u, v);
      }

      const geometry1 = new THREE.ExtrudeGeometry(shape, extrude_settings);
      geometry1.translate(0, 0, 0);
      const geometry2 = new THREE.ExtrudeGeometry(shape, extrude_settings);
      geometry2.translate(0, 0, (this.wall_height + this.window_height) / 2);
      geometries.push(geometry1);
      geometries.push(geometry2);
    }

    const geometry = BufferGeometryUtils.mergeBufferGeometries(geometries);
    const material = new THREE.MeshStandardMaterial({
      color: this.wall_color,
      // wireframe: true,
    });
    const object = new THREE.Mesh(geometry, material);
    object.position.set(-this.width / 2, 0, -this.height / 2);
    object.castShadow = true;
    object.receiveShadow = true;

    object.rotateX(Math.PI / 2);
    object.translateZ(-this.wall_height);

    this.objects.push(object);
  }

  createRoom(room_center) {
    const extrude_settings = {
      steps: 2,
      depth: 0.1,
      bevelEnabled: false,
    };

    let i = 0;
    const thresh = new THREE.Vector3(-this.width / 2, 0, -this.height / 2);
    for (const poly of this.room) {
      const center = room_center[i];
      const three_center = new THREE.Vector3(
        center[1] * this.width,
        (9 / 8) * this.wall_height,
        center[0] * this.height
      );
      three_center.add(thresh);

      this.room_centers.push(three_center);
      const pred = poly.pred;
      const shape = new THREE.Shape();
      let start = true;
      for (let [u, v] of poly.polygon) {
        u = u * this.width;
        v = v * this.height;

        if (start) {
          start = false;
          shape.moveTo(u, v);
        }
        shape.lineTo(u, v);
      }

      const geometry = new THREE.ExtrudeGeometry(shape, extrude_settings);
      geometry.translate(0, 0, this.wall_height - 0.1);

      const material = new THREE.MeshStandardMaterial({
        color: commonColors[pred].rgb,
        // wireframe: true,
      });
      const object = new THREE.Mesh(geometry, material);
      object.position.set(-this.width / 2, 0, -this.height / 2);
      object.castShadow = false;
      object.receiveShadow = true;

      object.rotateX(Math.PI / 2);
      object.translateZ(-this.wall_height);

      this.objects.push(object);
      i++;
    }
  }

  getSize() {
    return {
      width: this.bounding_box.bottom_right.x - this.bounding_box.top_left.x,
      height: this.bounding_box.bottom_right.y - this.bounding_box.top_left.y,
    };
  }

  addObject(object, polygon) {
    if (object == "wall") {
      this.wall.push(polygon);
    } else if (object == "window") {
      this.window.push(polygon);
    } else if (object == "door") {
      this.door.push(polygon);
    } else if (object == "room") {
      this.room.push(polygon);
    }
  }

  getDoors(room_number) {
    let ar = [];

    for (const door of this.door_objects) {
      if (
        parseInt(door.connection[0]) == parseInt(room_number) ||
        parseInt(door.connection[1]) == parseInt(room_number)
      ) {
        ar.push(door.object);
      }
    }
    return ar;
  }

  addOffset(x, y) {
    const addition = new THREE.Vector3(x, 0, y);
    for (const obj of this.objects) {
      obj.position.add(addition);
    }
    for (const coord of this.room_centers) {
      coord.add(addition);
    }
  }

  setMratio(m_ratio) {
    this.m_ratio = m_ratio;
  }

  setWHratio(wh_ratio) {
    this.wh_ratio = wh_ratio;
  }

  setHeight(height) {
    this.height = height;
    this.width = this.wh_ratio * height;
    // this.m_ratio = this.m_ratio * height;
    this.m_ratio = 3;
  }

  setWallHeight(height) {
    this.wall_height = height * this.m_ratio;
  }

  setDoorHeight(height) {
    this.door_height = height * this.m_ratio;
  }

  setWindowHeight(height) {
    this.window_height = height * this.m_ratio;
  }
}

const createFloor = (polygon) => {
  const {
    connection,
    wall,
    door,
    window,
    room,
    m_ratio,
    wh_ratio,
    room_center,
  } = polygon;

  console.log(polygon);

  const floor = new Floor();
  floor.setMratio(m_ratio);
  floor.setWHratio(wh_ratio);

  floor.setHeight(40);
  floor.setWallHeight(2);
  floor.setDoorHeight(1.7);
  floor.setWindowHeight(1);

  for (const w of wall) {
    floor.addObject("wall", w);
  }
  floor.createWall();

  for (const w of door) {
    floor.addObject("door", w);
  }
  floor.createDoor(connection);

  for (const w of window) {
    floor.addObject("window", w);
  }
  floor.createWindow();

  for (const w of room) {
    floor.addObject("room", w);
  }
  floor.createRoom(room_center);

  return floor;
};

export { createFloor };
