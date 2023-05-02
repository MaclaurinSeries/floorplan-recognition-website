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
  this.input_panel = document.getElementById(input_panel_id);

  this.load_screen = document.getElementById(load_screen_id);
  this.load_bar = document.getElementById(load_bar_id);
  this.load_info = document.getElementById(load_info_id);

  this.displayed_floor = 1;
  this.floor_cnt = 1;
  this.room_poly = {};

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

    const load_bar_promise = this.load_bar.animate(
      opacity_down,
      animation_timing
    ).finished;
    const load_info_promise = this.load_info.animate(
      opacity_down,
      animation_timing
    ).finished;

    await load_bar_promise, load_info_promise;

    this.in_progress = false;
    this.load_bar.style.width = "0%";

    this.load_bar.style.opacity = "0.0";
    this.load_info.style.opacity = "0.0";
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
      for (let i = 0; i < this.floor_cnt; i++) {
        let x = picked[i].map((e) => e[0]);
        let y = picked[i].map((e) => e[1]);
        this.crop_size.push([
          Math.min(...x),
          Math.min(...y),
          Math.max(...x),
          Math.max(...y),
        ]);
      }
    } else if (PID == "C0") {
      // ini hasil symbol detection
      const { name, preds } = data;
      for (let i = 0; i < this.floor_cnt; i++) {
        let picked = preds[i];

        const polygon = {};
        for (let pred of picked) {
          if (!(pred["names"] in polygon)) {
            polygon[pred["names"]] = [];
          }
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

      this.putImage(
        floor,
        "vectorization",
        this.image,
        null,
        {
          wall: [
            edges.map((x) => {
              return [
                [
                  coords[2 * x["vertices"][0] + 1],
                  coords[2 * x["vertices"][0]],
                ],
                [
                  coords[2 * x["vertices"][1] + 1],
                  coords[2 * x["vertices"][1]],
                ],
                x["thickness_meter"],
              ];
            }),
          ],
          door: [
            edges
              .map((x) => {
                let x1 = coords[2 * x["vertices"][0] + 1];
                let y1 = coords[2 * x["vertices"][0]];
                let x2 = coords[2 * x["vertices"][1] + 1];
                let y2 = coords[2 * x["vertices"][1]];

                let doors = x["doors"].map((door) => {
                  let xx = door["projection"] * (x2 - x1) + x1;
                  let yy = door["projection"] * (y2 - y1) + y1;

                  let vector = [x2 - x1, y2 - y1];

                  return [
                    [xx, yy],
                    vector,
                    door["length"],
                    x["thickness_meter"],
                  ];
                });

                return doors;
              })
              .filter((x) => x.length > 0)
              .map((x) => x[0]),
          ],
          window: [
            edges
              .map((x) => {
                let x1 = coords[2 * x["vertices"][0] + 1];
                let y1 = coords[2 * x["vertices"][0]];
                let x2 = coords[2 * x["vertices"][1] + 1];
                let y2 = coords[2 * x["vertices"][1]];

                let doors = x["windows"].map((door) => {
                  let xx = door["projection"] * (x2 - x1) + x1;
                  let yy = door["projection"] * (y2 - y1) + y1;

                  let vector = [x2 - x1, y2 - y1];
                  let norm = Math.sqrt(
                    vector[0] * vector[0] + vector[1] * vector[1]
                  );
                  vector[0] /= norm;
                  vector[1] /= norm;

                  return [
                    [xx, yy],
                    vector,
                    door["length"],
                    x["thickness_meter"],
                  ];
                });

                return doors;
              })
              .filter((x) => x.length > 0)
              .map((x) => x[0]),
          ],
        },
        this.crop_size[floor - 1],
        true,
        res["ppm"] / res["shape"][0]
      );
    } else if (PID == "D0") {
      const { name, floor, room_poly } = data;
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
      this.room_poly[floor] = rooms;
    } else if (PID == "E0") {
      const { name, floor, preds, labels } = data;

      this.labels = labels;
      const poly = {};

      for (let i = 0; i < preds.length; i++) {
        const obj = this.room_poly[floor][i].polygon;
        this.room_poly[floor][i].pred = preds[i];

        if (!poly[preds[i]]) {
          poly[preds[i]] = [];
        }

        poly[preds[i]].push(obj);
      }

      this.putImage(
        floor,
        "room-classification",
        this.image,
        null,
        poly,
        this.crop_size[floor - 1]
      );
    }
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

  this.putImage = (
    floor,
    name,
    image = null,
    fill = null,
    polygon = null,
    crop = null,
    line_independence = false,
    ppm = null
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

          let fill_color = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.3)`;
          let stroke_color = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

          for (let i = 0; i < polygon[cls].length; i++) {
            const poly = polygon[cls][i];

            if (line_independence) {
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
            } else {
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
              ctx.strokeStyle = stroke_color;
              ctx.fillStyle = fill_color;
              ctx.lineWidth = 3;
              ctx.fill();
              ctx.stroke();
            }
          }
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
