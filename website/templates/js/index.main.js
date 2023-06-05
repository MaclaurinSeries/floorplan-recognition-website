const rooms_label = [
  "Outdoor",
  "Kitchen",
  "Dining",
  "Bedroom",
  "Bath",
  "Entry",
  "Storage",
  "Garage",
  "Room",
  "LivingRoom",
];
const icons_label = [
  "Door",
  "Window",
  "Closet",
  "ElectricalAppliance",
  "Toilet",
  "Sink",
  "SaunaBench",
  "Fireplace",
  "Bathtub",
  "Chimney",
  "Stairs",
];

let FloorplanRecognition = function (ID, current_time, components) {
  const {
    input_file_id,
    input_help_id,
    input_panel_id,
    text_holder_id,
    button_submit_id,
    load_screen_id,
    load_bar_id,
    load_info_id,
    button_3d_model,
  } = components;

  this.ID = ID;
  this.current_time = current_time;
  this.image = null;
  this.is_valid_image = false;

  this.first_state = true;
  this.in_progress = false;

  this.input_info = document.getElementById(input_help_id);
  this.input_file = document.getElementById(input_file_id);
  this.text_holder = document.getElementById(text_holder_id);
  this.button_submit = document.getElementById(button_submit_id);
  this.button_3d_model = document.getElementById(button_3d_model);
  this.input_panel = document.getElementById(input_panel_id);

  this.load_screen = document.getElementById(load_screen_id);
  this.load_bar = document.getElementById(load_bar_id);
  this.load_info = document.getElementById(load_info_id);

  this.displayed_floor = 1;
  this.floor_cnt = 1;
  this.room_poly = {};

  this.polygon = [];
  this.m_ratio = [];
  this.graph = {};

  this.input_file.onchange = () => {
    let filename = String(this.input_file.value);
    let ext = filename.substring(filename.lastIndexOf(".") + 1).toLowerCase();

    if (
      this.input_file.files &&
      this.input_file.files[0] &&
      (ext == "png" || ext == "jpg")
    ) {
      let reader = new FileReader();

      reader.onload = (e) => {
        this.image = e.target.result;
        this.is_valid_image = true;
      };

      reader.readAsDataURL(this.input_file.files[0]);
    } else {
      this.image = null;
      this.is_valid_image = false;
    }
  };

  this.button_3d_model.onclick = async () => {
    if (this.polygon.length <= 0) {
      return;
    }
    if (this.in_progress) {
      return;
    }
    const form_data = new FormData();
    console.log(this.polygon);
    form_data.append("polygon", JSON.stringify(this.polygon));
    form_data.append("graph", JSON.stringify(this.graph));
    // console.log(body_data);
    fetch("/", {
      method: "POST",
      body: form_data,
    })
      .then((res) => {
        return res.text();
      })
      .then((res) => {
        let script;
        let win = window.open("");
        win.document.body.outerHTML = res;
        win.document.title = "3D Model";

        let link = document.createElement("link");
        link.rel = "stylesheet";
        link.href =
          "https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css";
        link.integrity =
          "sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm";
        link.crossOrigin = "anonymous";
        win.document.head.appendChild(link);

        script = win.document.createElement("script");
        script.innerHTML = `window.__polygon = Object.freeze(JSON.parse('${JSON.stringify(
          this.polygon
        )}'));
                            window.__graph = Object.freeze(JSON.parse('${JSON.stringify(
                              this.graph
                            )}'));`;
        script.id = "simulation-data";
        win.document.body.appendChild(script);

        for (let link of [
          "https://code.jquery.com/jquery-3.2.1.slim.min.js",
          "https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js",
          "https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js",
        ]) {
          script = win.document.createElement("script");
          win.document.body.appendChild(script);
          script.type = "text/javascript";
          script.src = link;
        }

        script = win.document.createElement("script");
        win.document.body.appendChild(script);
        script.type = "module";
        script.addEventListener("load", function (e) {
          win.initialization(win.__polygon, win.__graph);
        });
        script.src = `http://${window.location.host}/static/js/three.main.js`;
      });
  };

  this.button_submit.onclick = async () => {
    if (!this.is_valid_image) {
      return;
    }
    if (this.in_progress) {
      return;
    }
    this.in_progress = true;

    const opacity_up = [{ opacity: "0.0" }, { opacity: "1.0" }];
    const opacity_down = [{ opacity: "1.0" }, { opacity: "0.0" }];

    const animation_timing = {
      duration: 300,
      iterations: 1,
    };

    this.changeState(opacity_up, opacity_down, animation_timing);

    this.load_bar.style.opacity = "1.0";
    this.load_info.style.opacity = "1.0";

    for (const img of document.getElementsByTagName("img")) {
      img.src = "";
    }
    await this.sendImage();
  };

  this.setLoadStatus = async (load_info) => {
    const { percentage, info } = load_info;
    const current = this.load_bar.style.width;
    const next = String(percentage) + "%";

    const stretch_bar = [{ width: current }, { width: next }];
    const animation_timing = {
      duration: 300,
      iterations: 1,
      easing: "ease",
    };

    this.load_info.innerHTML = info;
    this.load_bar.style.width = next;
    await this.load_bar.animate(stretch_bar, animation_timing).finished;
  };

  this.changeState = async (opacity_up, opacity_down, animation_timing) => {
    this.opacity_up = opacity_up;
    this.opacity_down = opacity_down;
    this.animation_timing = animation_timing;
    if (!this.first_state) return;

    this.first_state = false;

    this.load_screen.setAttribute(
      "style",
      "visibility: visible; display: inline;"
    );
    await this.load_screen.animate(opacity_up, animation_timing).finished;

    const map_obj = {
      "row-span-2": "",
      "w-sm": "w-full",
      "grid-rows-2": "grid-rows-1",
      "my-auto": "my-4",
    };

    let class_name = this.input_panel.className;
    class_name = class_name.replace(
      /((row\-span\-2)|(w\-sm)|(grid\-rows\-2)|(my\-auto))/g,
      (match) => {
        return map_obj[match];
      }
    );

    this.input_panel.className = class_name;

    this.text_holder.style.display = "none";
    this.text_holder.style.visibility = "visible";

    document.getElementById("panel-holder").style.visibility = "visible";
    document.getElementById("panel-holder").style.display = "inline";

    await this.load_screen.animate(opacity_down, animation_timing).finished;
    this.load_screen.setAttribute(
      "style",
      "visibility: hidden; display: none;"
    );
  };

  this.sendImage = async () => {
    const endpoint = `ws://${window.location.host}/predict`;
    const socket = new WebSocket(endpoint);

    socket.addEventListener("open", (event) => {
      console.log("[open] Connection established");
      console.log(`Sending image to server ${endpoint}`);

      this.putImage(0, "original-image", this.image);

      socket.send(
        JSON.stringify({
          ID: this.ID,
          gnn: "graph-sage-jk",
          image_data: this.image,
        })
      );
    });

    socket.addEventListener("message", (event) => {
      const { PID, percentage, info, data } = JSON.parse(event.data);
      this.messageHandler(PID, percentage, info, data);
      this.setLoadStatus({ percentage: percentage, info: info });
    });

    socket.addEventListener("close", (event) => {
      this.setLoadStatus({ percentage: 100, info: "finished" });
    });
  };

  this.setFloor = (floor) => {
    floor = parseInt(floor);

    for (let i = 2; i <= this.floor_cnt; i++) {
      document.getElementById(`btn-Floor-${i}`).remove();
      document.getElementById(`Floor-${i}`).remove();
    }
    for (const img in document.getElementsByTagName("img")) {
      img.src = "";
    }
    for (let i = 2; i <= floor; i++) {
      const btn = document.getElementById(`btn-Floor-1`);
      const div = document.getElementById(`Floor-1`);

      const div_clone = div.cloneNode(true);
      const btn_clone = btn.cloneNode(true);

      div_clone.setAttribute("id", `Floor-${i}`);
      btn_clone.setAttribute("id", `btn-Floor-${i}`);
      btn_clone.innerHTML = `Floor-${i}`;

      div.parentNode.appendChild(div_clone);
      btn.parentNode.appendChild(btn_clone);
    }

    for (let i = 1; i <= floor; i++) {
      document
        .getElementById(`btn-Floor-${i}`)
        .addEventListener("click", this.changeFloor(i));
    }
    this.floor_cnt = floor;
    this.changeFloor(1)();
  };

  this.changeFloor = (floor_idx) => {
    return (event) => {
      for (let i = 1; i <= this.floor_cnt; i++) {
        const div = document.getElementById(`Floor-${i}`);
        const btn = document.getElementById(`btn-Floor-${i}`);
        if (i === floor_idx) {
          btn.style.border = "1px solid white";
          div.style.display = "inline";
        } else {
          btn.style.border = "none";
          div.style.display = "none";
        }
      }
    };
  };

  this.endLoadingBar = async () => {
    const load_bar_promise = this.load_bar.animate(
      this.opacity_down,
      this.animation_timing
    ).finished;
    const load_info_promise = this.load_info.animate(
      this.opacity_down,
      this.animation_timing
    ).finished;

    await load_bar_promise, load_info_promise;

    this.load_bar.style.width = "0%";

    this.load_bar.style.opacity = "0.0";
    this.load_info.style.opacity = "0.0";
  };

  this.messageHandler = (PID, percentage, info, data) => {
    const image_pid = ["C1", "C2", "D1"];

    if (image_pid.includes(PID)) {
      const { name, floor, image } = data;
      this.putImage(floor, name, image);
    } else if (PID == "A1") {
      // ini hasil roi detection
      const { name, floor, preds } = data;
      this.setFloor(floor);

      let picked = preds.map((x) => x["polygon_n"]);

      const polygon = {
        roi: picked,
      };

      this.putImage(0, "roi-detection", this.image, null, polygon);

      this.crop_size = [];
      this.polygon = [];
      for (let i = 0; i < this.floor_cnt; i++) {
        let x = picked[i].map((e) => e[0]);
        let y = picked[i].map((e) => e[1]);
        this.crop_size.push([
          Math.min(...x),
          Math.min(...y),
          Math.max(...x),
          Math.max(...y),
        ]);
        this.polygon.push({});
        this.m_ratio.push(-1);
      }
    } else if (PID == "C0") {
      const { name, preds } = data;
      for (let i = 0; i < this.floor_cnt; i++) {
        let picked = preds[i];

        const polygon = {};
        for (let name of icons_label) {
          polygon[name] = [];
        }
        for (let pred of picked) {
          let bboxn = pred["bounding_box_n"];
          let bbox = [
            [bboxn[0], bboxn[1]],
            [bboxn[2], bboxn[1]],
            [bboxn[2], bboxn[3]],
            [bboxn[0], bboxn[3]],
          ];
          polygon[pred["names"]].push(bbox);
        }

        this.putImage(
          i + 1,
          "symbol-detection",
          this.image,
          null,
          polygon,
          this.crop_size[i]
        );
      }
    } else if (PID == "C3") {
      const { name, floor, coords, edges, res } = data;

      const wall = edges.reduce((next, prev) => {
        return next.concat(prev["polygon"]);
      }, []);
      const door = edges.reduce((next, prev) => {
        return next.concat(prev["doors"].map((x) => x["polygon"]));
      }, []);
      const window = edges.reduce((next, prev) => {
        return next.concat(prev["windows"].map((x) => x["polygon"]));
      }, []);
      this.m_ratio[floor - 1] = res["ppm"] / res["shape"][0];

      this.putImage(
        floor,
        "vectorization",
        this.image,
        null,
        {
          "0_wall": wall,
          "1_door": door,
          "2_window": window,
        },
        this.crop_size[floor - 1],
        "full",
        this.m_ratio[floor - 1]
      );

      this.polygon[floor - 1]["wall"] = wall;
      this.polygon[floor - 1]["door"] = door;
      this.polygon[floor - 1]["window"] = window;
      this.polygon[floor - 1]["m_ratio"] = this.m_ratio[floor - 1];
      this.polygon[floor - 1]["wh_ratio"] =
        (this.crop_size[floor - 1][2] - this.crop_size[floor - 1][0]) /
        (this.crop_size[floor - 1][3] - this.crop_size[floor - 1][1]);
    } else if (PID == "D0") {
      const { name, floor, room_poly, x_location, edge, door_edges } = data;

      const rooms = [];
      for (const room of room_poly) {
        const transpose = [];
        for (let i = 0; i < room[0].length; i++) {
          transpose.push([room[1][i], room[0][i]]);
        }
        rooms.push({
          polygon: transpose,
          pred: -1,
        });
      }
      this.room_poly[floor - 1] = rooms;
      this.graph = {
        x: x_location,
        edge: edge,
      };

      const door = door_edges.reduce((next, prev) => {
        return next.concat(prev["doors"].map((x) => x["polygon"]));
      }, []);
      const connection = door_edges.reduce((next, prev) => {
        return next.concat(prev["doors"].map((x) => x["connection"]));
      }, []);
      this.polygon[floor - 1]["door"] = door;
      this.polygon[floor - 1]["connection"] = connection;
      this.polygon[floor - 1]["room_center"] = x_location;

      const poly = {};
      poly["room"] = [];
      for (let rpl of rooms) {
        poly["room"].push(rpl.polygon);
      }

      this.putImage(
        floor,
        "graph-construction",
        this.image,
        null,
        poly,
        this.crop_size[floor - 1],
        "graph",
        null,
        this.graph
      );
    } else if (PID == "D2") {
      const { name, floor, preds } = data;
      let picked = preds;

      const polygon = {};
      for (let pred of picked) {
        if (!("text" in polygon)) {
          polygon["text"] = [];
        }
        let bboxn = pred["bbox"];
        polygon["text"].push(bboxn);
      }

      this.putImage(
        floor,
        "text-detection",
        this.image,
        null,
        polygon,
        this.crop_size[floor - 1]
      );
    } else if (PID == "E0") {
      const { name, floor, preds, labels } = data;

      this.labels = labels;
      const poly = {};

      for (let room of rooms_label) {
        poly[room] = [];
      }
      for (let i = 0; i < preds.length; i++) {
        const obj = this.room_poly[floor - 1][i].polygon;
        this.room_poly[floor - 1][i].pred = preds[i];

        poly[rooms_label[preds[i]]].push(obj);
      }

      this.polygon[floor - 1]["room"] = this.room_poly[floor - 1];

      this.putImage(
        floor,
        "room-classification",
        this.image,
        null,
        poly,
        this.crop_size[floor - 1]
      );

      if (floor == this.floor_cnt) {
        this.in_progress = false;
        this.endLoadingBar();
      }
    }
  };

  this.putImage = (
    floor,
    name,
    image = null,
    fill = null,
    polygon = null,
    crop = null,
    mode = "transparent",
    ppm = null,
    graph = null
  ) => {
    let img = document.createElement("img");
    img.onload = (e) => {
      const vw = window.innerWidth;
      const vw40 = 0.4 * vw;
      let r = img.width / img.height;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      canvas.width = vw40;
      canvas.height = vw40;

      let neww, newh;
      if (img.width > img.height) {
        neww = vw40;
        newh = vw40 / r;
      } else {
        neww = vw40 * r;
        newh = vw40;
      }
      let stx = (vw40 - neww) / 2,
        sty = (vw40 - newh) / 2;
      ppm = newh * ppm;

      ctx.fillStyle = "rgb(127, 127, 127)";
      if (fill !== null) {
        ctx.fillStyle = fill;
      }
      ctx.fillRect(0, 0, vw40, vw40);

      let x1 = 0,
        y1 = 0,
        x2 = img.width,
        y2 = img.height;
      if (crop !== null) {
        [x1, y1, x2, y2] = crop;
        x1 = x1 * img.width;
        y1 = y1 * img.height;
        x2 = x2 * img.width;
        y2 = y2 * img.height;
        r = (x2 - x1) / (y2 - y1);
        if (x2 - x1 > y2 - y1) {
          neww = vw40;
          newh = vw40 / r;
        } else {
          neww = vw40 * r;
          newh = vw40;
        }
        stx = (vw40 - neww) / 2;
        sty = (vw40 - newh) / 2;
      }
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, x1, y1, x2 - x1, y2 - y1, stx, sty, neww, newh);

      if (polygon !== null) {
        let color_index = 0;
        let first_clear = true;
        for (const cls in polygon) {
          const rgb = commonColors[color_index];
          color_index++;

          let fill_color, stroke_color;
          if (mode == "transparent" || mode == "graph") {
            fill_color = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.3)`;
            stroke_color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          } else if (mode == "full") {
            fill_color = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 1.0)`;
            stroke_color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0)`;
            if (cls == "0_wall") {
              fill_color = `rgba(0,0,0, 1.0)`;
            }
          }

          for (let i = 0; i < polygon[cls].length; i++) {
            const poly = polygon[cls][i];

            if (mode == "line") {
              if (first_clear) {
                ctx.fillStyle = "rgb(127, 127, 127)";
                ctx.fillRect(0, 0, vw40, vw40);
                first_clear = false;
              }
              if (poly[0].length == 3) {
                for (let [u, v, w] of poly) {
                  u[0] = u[0] * neww + stx;
                  u[1] = u[1] * newh + sty;
                  v[0] = v[0] * neww + stx;
                  v[1] = v[1] * newh + sty;

                  ctx.beginPath();
                  ctx.moveTo(u[0], u[1]);
                  ctx.lineTo(v[0], v[1]);
                  ctx.closePath();
                  ctx.strokeStyle = "#ffffff";
                  ctx.lineWidth = w * ppm;
                  ctx.stroke();
                }
              } else {
                for (let [u, v, l, w] of poly) {
                  u[0] = u[0] * neww + stx;
                  u[1] = u[1] * newh + sty;

                  v[0] *= neww;
                  v[1] *= newh;
                  let norm = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
                  v[0] /= norm;
                  v[1] /= norm;

                  let half = (l * ppm) / 2;

                  ctx.beginPath();
                  ctx.moveTo(u[0] - half * v[0], u[1] - half * v[1]);
                  ctx.lineTo(u[0] + half * v[0], u[1] + half * v[1]);
                  ctx.closePath();
                  ctx.strokeStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
                  ctx.lineWidth = w * ppm;
                  ctx.stroke();
                }
              }
            } else if (
              mode == "transparent" ||
              mode == "full" ||
              mode == "graph"
            ) {
              if (first_clear && mode == "full") {
                ctx.fillStyle = "rgb(127, 127, 127)";
                ctx.fillRect(0, 0, vw40, vw40);
                first_clear = false;
              }
              let first = true;
              ctx.beginPath();
              for (let [u, v] of poly) {
                u = u * neww + stx;
                v = v * newh + sty;
                if (first) {
                  first = false;
                  ctx.moveTo(u, v);
                } else {
                  ctx.lineTo(u, v);
                }
              }
              ctx.closePath();
              ctx.fillStyle = fill_color;
              ctx.fill();
              if (mode != "full") {
                ctx.strokeStyle = stroke_color;
                ctx.lineWidth = 3;
                ctx.stroke();
              }
            }
          }
        }
      }
      if (mode == "graph") {
        const { x, edge } = graph;
        console.log(x, edge);
        for (let i = 0; i < edge[0].length; i++) {
          const first = x[edge[0][i]];
          const second = x[edge[1][i]];

          ctx.beginPath();
          ctx.moveTo(first[1] * neww + stx, first[0] * newh + sty);
          ctx.lineTo(second[1] * neww + stx, second[0] * newh + sty);
          ctx.closePath();

          ctx.strokeStyle = "blue";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        for (const loc of x) {
          ctx.beginPath();
          ctx.arc(
            loc[1] * neww + stx,
            loc[0] * newh + sty,
            7,
            2 * Math.PI,
            false
          );
          ctx.closePath();
          ctx.fillStyle = "green";
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = "#003300";
          ctx.stroke();
        }
      }

      if (floor > 0) {
        const div = document.getElementById(`Floor-${floor}`);
        div.getElementsByClassName(name)[0].src = canvas.toDataURL();
      } else {
        document.getElementsByName(name)[0].src = canvas.toDataURL();
      }
    };
    img.src = image;
  };
};

const commonColors = [
  [255, 0, 0], // Red
  [0, 255, 0], // Green
  [0, 0, 255], // Blue
  [255, 255, 0], // Yellow
  [255, 165, 0], // Orange
  [128, 0, 128], // Purple
  [255, 192, 203], // Pink
  [165, 42, 42], // Brown
  [128, 128, 128], // Gray
  [128, 0, 0], // Maroon
  [0, 128, 0], // Dark Green
  [0, 128, 128], // Teal
  [0, 0, 128], // Navy
  [139, 69, 19], // Saddle Brown
  [255, 20, 147], // Deep Pink
  [255, 105, 180], // Hot Pink
  [70, 130, 180], // Steel Blue
  [32, 178, 170], // Light Sea Green
  [100, 149, 237], // Cornflower Blue
  [255, 99, 71], // Tomato
  [255, 215, 0], // Gold
  [0, 255, 255], // Cyan
  [238, 130, 238], // Violet
];

document.body.onload = () => {
  new FloorplanRecognition(0, Date.now(), {
    input_file_id: "file-input",
    input_help_id: "file-input-help",
    input_panel_id: "panel-submit",
    button_submit_id: "submit-image",
    load_screen_id: "load-screen",
    load_bar_id: "loading-bar",
    load_info_id: "loading-info",
    text_holder_id: "text-holder",
    button_3d_model: "btn-3d-model",
  });
};

function shuffleArray(array) {
  for (var i = array.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}
