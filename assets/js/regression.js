// globals
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.font = "12px Arial";
ctx.strokeStyle = "white";
ctx.fillStyle = "white";
let isDragging = false;
let selectedDataPoint;

canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("touchstart", handleMouseDown);
canvas.addEventListener("mousemove", handleMouseMove);
canvas.addEventListener("touchmove", handleMouseMove);
canvas.addEventListener("mouseup", handleMouseUp);
canvas.addEventListener("touchend", handleMouseUp);

// A DataPoint
class DataPoint {
    constructor(x, y) {
      this.x = x;
      this.y = y;
      this.r = 5;
      this.dragged = false;
    }

    draw() {
        let lastColor = ctx.fillStyle;
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.ellipse(this.x, this.y, this.r, this.r, 0, 0, 360);
        ctx.fill();
        ctx.fillStyle = lastColor;
    }
}

class Line {
    constructor(angle) {
        this.angle = angle;
    }

    draw() {
        let lastColor = ctx.strokeStyle;
        let lastWidth = ctx.lineWidth;
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 3;
        let x = 150;
        let y = 150;
        let r = 300;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + r * Math.cos(this.angle), y + r * Math.sin(this.angle));
        ctx.moveTo(x, y);
        ctx.lineTo(x + r * Math.cos(this.angle + Math.PI), y + r * Math.sin(this.angle + Math.PI));
        ctx.stroke();
        ctx.strokeStyle = lastColor;
        ctx.lineWidth = lastWidth;
    }
}

let dataPoints = [
    new DataPoint(280, 60),
    new DataPoint(200, 90),
    new DataPoint(250, 140),
    new DataPoint(160, 130)
];

let line = new Line(fitLine());

function drawDataPoints() {
    // datapoint
    dataPoints.forEach(dp => {
        dp.draw();
    });
}

function handleMouseDown(event) {
  dataPoints.forEach((dp) => {
    if (isInsideCircle(event.offsetX, event.offsetY, dp)) {
      isDragging = true;
      selectedDataPoint = dp;
    }
  });
}

function handleMouseMove(event) {
    if (isDragging) {
      selectedDataPoint.x = event.offsetX;
      selectedDataPoint.y = event.offsetY;
      draw();
    }
}

function handleMouseUp(event) {
    isDragging = false;
    line.angle = fitLine();
    draw();
}

function isInsideCircle(x, y, circle) {
    const dx = x - circle.x;
    const dy = y - circle.y;
    return dx * dx + dy * dy <= circle.r * circle.r;
}

function drawAxes() {
    ctx.beginPath();
    ctx.moveTo(150, 300);
    ctx.lineTo(150, 0);
    ctx.lineTo(146, 4);
    ctx.lineTo(154, 4);
    ctx.lineTo(150, 0);
    ctx.stroke();
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(0, 150);
    ctx.lineTo(300, 150);
    ctx.lineTo(296, 146);
    ctx.lineTo(296, 154);
    ctx.lineTo(300, 150);
    ctx.stroke();
    ctx.fill();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();
    line.draw();
    drawDataPoints();
    ctx.fillText("Move the data to fit a line", 164, 300);
}

function fitLine() {
    let xs = [];
    let ys = [];
    
    dataPoints.forEach(dp => {
        xs.push(dp.x - 150);
        ys.push(-(dp.y - 150));
    })

    let num = 0
    let den = 0

    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;

    for (let i = 0; i < xs.length; i++) {
        num += xs[i] * ys[i]
        den += xs[i] * xs[i]
    }

    if (den === 0) {
        den = 0.01
    }

    let res = num / den;
    let angle = Math.atan(-res);

    return angle;
}

draw();