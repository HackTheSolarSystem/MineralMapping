import { Component } from '@angular/core';

@Component({
  selector: 'overlay-grid',
  templateUrl: './grid.component.html',
  styleUrls: ['./grid.component.scss'],
})
export class GridComponent {
  public testArray = [
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 4],
    [1, 2, 3, 5, 5, 6, 7, 8, 9, 4, 4],
    [1, 2, 5, 5, 5, 6, 7, 8, 9, 4, 4],
  ];
}
