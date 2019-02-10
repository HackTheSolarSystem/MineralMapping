import { Component } from '@angular/core';

import data from '../../constants/obj2_array.json';

@Component({
  selector: 'overlay-grid',
  templateUrl: './grid.component.html',
  styleUrls: ['./grid.component.scss'],
})
export class GridComponent {
  constructor() {
    console.log('data in grid is:', data);
  }
  public realData = data ? data.data : [];
  public testArray = [
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 5],
    [1, 2, 3, 4, 6, 6, 7, 8, 9, 5, 4],
    [1, 2, 3, 5, 5, 6, 7, 8, 9, 4, 4],
    [1, 2, 5, 5, 5, 6, 7, 8, 9, 4, 4],
  ];
}
