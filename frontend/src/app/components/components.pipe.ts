import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'getColor',
})
export class GetColorPipe implements PipeTransform {
  private colorMap = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue',
    6: 'indigo',
    7: 'violet',
    8: 'grey',
    9: 'brown',
  };
  transform(number) {
    return this.colorMap[number];
  }
}
