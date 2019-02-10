import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { DetailViewComponent } from './views/detail-view/detail-view.component';
import { SummaryViewComponent } from './views/summary-view/summary-view.component';
import { ComponentsModule } from './components/components.module';

@NgModule({
  declarations: [
    AppComponent,
    DetailViewComponent,
    SummaryViewComponent,
  ],
  imports: [
    BrowserModule,
    ComponentsModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
