import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'getColor',
})
export class GetColorPipe implements PipeTransform {
  private colorMap = {
    "Troilite": 'red',
    "Taenite": 'orange',
    "Pyroxene": 'yellow',
    "Pentlandite": 'green',
    "Olivine": 'blue',
    "Millerite": 'indigo',
    "Kamacite": 'violet',
  };
  transform(number) {
    return this.colorMap[number];
  }
}
