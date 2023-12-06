// globals
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let height = 200;
let width = 500;
ctx.font = "12px Monospace"
if (window.innerWidth < 480) {
    canvas.width = 300;
    width = 300;
    ctx.font = "8px Monospace"
}
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
        this.r = 8;
        this.dragged = false;
    }

    draw() {
        let lastColor = ctx.fillStyle;
        ctx.fillStyle = "orange";
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
        ctx.strokeStyle = "white";
        ctx.lineWidth = 5;
        let x = width / 2;
        let y = height / 2;
        let r = width;
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
    new DataPoint(width / 2 + 20, height / 2 - 20),
    new DataPoint(width / 2 + 40, height / 2 - 40),
    new DataPoint(width / 2 + 100, height / 2 - 10),
    new DataPoint(width / 2 + 120, height / 2 - 80)
];

let line = new Line(fitLine());

function drawDataPoints() {
    // datapoint
    dataPoints.forEach(dp => {
        dp.draw();
    });
}

function handleMouseDown(event) {
    event.preventDefault();
    if (event.type === "mousedown") {
        var offsetX = event.offsetX;
        var offsetY = event.offsetY;
    } else if (event.type === "touchstart") {
        var touch = event.targetTouches[0];
        var offsetX = touch.clientX - event.target.offsetLeft;
        var offsetY = touch.clientY - event.target.offsetTop;
    }
    dataPoints.forEach((dp) => {
        if (isInsideCircle(offsetX, offsetY, dp)) {
            isDragging = true;
            selectedDataPoint = dp;
        }
    });
}

function handleMouseMove(event) {
    event.preventDefault();
    if (event.type === "mousemove") {
        var offsetX = event.offsetX;
        var offsetY = event.offsetY;
    } else if (event.type === "touchmove") {
        var touch = event.targetTouches[0];
        var offsetX = touch.clientX - event.target.offsetLeft;
        var offsetY = touch.clientY - event.target.offsetTop;
    }

    let isOverDataPoint = false;
    dataPoints.forEach((dp) => {
        if (isInsideCircle(offsetX, offsetY, dp)) {
            isOverDataPoint = true;
        }
    });
    canvas.style.cursor = isOverDataPoint ? "pointer" : "default";

    if (isDragging) {
        offsetX = Math.max(0, Math.min(canvas.width, offsetX));
        offsetY = Math.max(0, Math.min(canvas.height, offsetY));
        selectedDataPoint.x = offsetX;
        selectedDataPoint.y = offsetY;
        draw();
    }
}

function handleMouseUp(event) {
    event.preventDefault();
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
    ctx.moveTo(width / 2, height);
    ctx.lineTo(width / 2, 0);
    ctx.lineTo(width / 2 - 4, 4);
    ctx.lineTo(width / 2 + 4, 4);
    ctx.lineTo(width / 2, 0);
    ctx.stroke();
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.lineTo(width - 4, height / 2 - 4);
    ctx.lineTo(width - 4, height / 2 + 4);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    ctx.fill();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();
    line.draw();
    drawDataPoints();
}

function fitLine() {
    let xs = [];
    let ys = [];

    dataPoints.forEach(dp => {
        xs.push(dp.x - width / 2);
        ys.push(-(dp.y - height / 2));
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