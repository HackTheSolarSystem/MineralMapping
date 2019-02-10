import {Component, ElementRef, HostListener, OnInit, ViewChild} from '@angular/core';
import {container} from '@angular/core/src/render3/instructions';
import * as d3 from 'd3';

@Component({
  selector: 'overlay-grid',
  templateUrl: './grid.component.html',
  styleUrls: ['./grid.component.scss'],
})
export class GridComponent {
  @ViewChild('container') container: ElementRef;
  @ViewChild('source') image: ElementRef;
  public testArray = [
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 4],
    [1, 2, 3, 5, 5, 6, 7, 8, 9, 4, 4],
    [1, 2, 5, 5, 5, 6, 7, 8, 9, 4, 4],
  ];

  ngOnInit() {
    const canvas = d3.select("#img-zoom-container").append('svg').attr('width', '514px').attr('height', '514px');


    const width = 514;
    const height = 514;
    const arrayWidth = this.testArray[0].length;
    const arrayLength = this.testArray.length;
    const widthMultipler = width / arrayWidth;
    const heightMultipler = height / arrayLength;

    let widthCount = 0;
    let heightCount = 0;
    //
    //
    for (let subArray of this.testArray) {
      for (let value of subArray) {

        // console.log('moo', widthCount, heightCount, , widthMultipler)
        canvas.append('rect')
          .attr('x', widthCount)
          .attr('y', heightCount)
          .attr('width', widthMultipler)
          .attr('height', heightMultipler)
          .style('fill','red');
        widthCount = widthCount + widthMultipler;

        if (widthCount > width) {
          widthCount = 0;
        }
      }
      heightCount = heightCount + heightMultipler;
    }

  }

  imageZoom(imgID, gridContainerId, resultID) {
    function moveLens(e) {
      var pos, x, y;
      /*prevent any other actions that may occur when moving over the image:*/
      e.preventDefault();
      /*get the cursor's x and y positions:*/
      pos = getCursorPos(e);
      /*calculate the position of the lens:*/
      x = pos.x - (lens.offsetWidth / 2);
      y = pos.y - (lens.offsetHeight / 2);
      /*prevent the lens from being positioned outside the image:*/
      if (x > img.width - lens.offsetWidth) { x = img.width - lens.offsetWidth; }
      if (x < 0) { x = 0; }
      if (y > img.height - lens.offsetHeight) { y = img.height - lens.offsetHeight; }
      if (y < 0) { y = 0; }
      /*set the position of the lens:*/
      lens.style.left = x + "px";
      lens.style.top = y + "px";
      /*display what the lens "sees":*/
      result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
    }
    function getCursorPos(e) {
      var a, x = 0, y = 0;
      e = e || window.event;
      /*get the x and y positions of the image:*/
      a = img.getBoundingClientRect();
      /*calculate the cursor's x and y coordinates, relative to the image:*/
      x = e.pageX - a.left;
      y = e.pageY - a.top;
      /*consider any page scrolling:*/
      x = x - window.pageXOffset;
      y = y - window.pageYOffset;
      return { x: x, y: y };
    }

    var img, lens, result, cx, cy, gridContainer;
    img = document.getElementById(imgID);
    result = document.getElementById(resultID);
    gridContainer = document.getElementById(gridContainerId);

    /*create lens:*/
    lens = document.createElement("DIV");
    lens.setAttribute("class", "img-zoom-lens");
    /*insert lens:*/
    img.parentElement.insertBefore(lens, img);
    /*calculate the ratio between result DIV and lens:*/
    cx = result.offsetWidth / lens.offsetWidth;
    cy = result.offsetHeight / lens.offsetHeight;

    console.log('cs', cx, cy)
    /*set background properties for the result DIV:*/
    result.style.backgroundImage = "url('" + img.src + "')";
    result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
    /*execute a function when someone moves the cursor over the image, or the lens:*/
    lens.addEventListener("mousemove", moveLens);
    img.addEventListener("mousemove", moveLens);
    /*and also for touch screens:*/
    lens.addEventListener("touchmove", moveLens);
    img.addEventListener("touchmove", moveLens);
  }
}
