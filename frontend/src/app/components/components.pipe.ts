import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'getColor',
})
export class GetColorPipe implements PipeTransform {
  private colorMap = {
    "Troilite": '#4D79A8',
    "Taenite": '#F28E2C',
    "Pyroxene": '#E05758',
    "Pentlandite": '#76B7B2',
    "Olivine": '#58A14E',
    "Millerite": '#EDC949',
    "Kamacite": '#B07AA1',
  };
  transform(number) {
    return this.colorMap[number];
  }
}
