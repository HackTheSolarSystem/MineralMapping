import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { GridComponent } from './grid/grid.component';

import { GetColorPipe } from './components.pipe';

@NgModule({
  imports: [
    BrowserModule,
  ],
  declarations: [
    GridComponent,
    GetColorPipe,
  ],
  exports: [GridComponent],

})
export class ComponentsModule {
}
