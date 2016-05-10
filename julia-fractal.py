from gi.repository import Gtk, Gdk, GdkPixbuf

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

import matplotlib.pyplot as pl
import numpy as np
import time

from numba import jit, vectorize, complex128, float64, int32



def two_point_interp(xp, x1, x2, y1, y2):
    '''
    A function that maps the xp value from the x1..x2 range 
    to the value in the y1..y2 range.
    '''
    return np.interp(xp, [x1, x2], [y1, y2])


def complex_grid(xlim, ylim, nx, ny):
    '''
    returns a nx x ny grid of complex numbers
    bounded by xlim and ylim ranges.
    '''
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    xx, yy = np.meshgrid(x, y) 

    return xx + 1j*yy


@jit(int32(complex128, complex128, int32, float64, int32), nopython=True, cache=True)
def iterate(z, C, n, zmax, niter):
    '''
    Return the number of iteration 
    needed for the absolute value 
    of z to become greater than zmax
    '''
    zmax2 = zmax**2
    for k in range(niter):
        
        z = pow(z, n) + C
        if z.imag**2 + z.real**2 > zmax2:
            break
            
    return k


@vectorize([int32(complex128, complex128, int32, float64, int32)], target='parallel', nopython=True)
def fractal(zz, C, n, zmax, niter):   

    return iterate(zz, C, n, zmax, niter)
    


class App:

    def __init__(self):
    
        self.xsize, self.ysize = 600, 600
        
        self.xmin, self.xmax = -1.5, 1.5
        self.ymin, self.ymax = -1.5, 1.5
        
        self.x, self.y = (-0.4, 0.6)
    
        self.n     = 2
        self.zmax  = 4.0
        self.niter = 256
        
        self.dpi  = 100
        self.cmap = 'Set3'
        
        self.digits = 12
        
        self.entries = {}
        
        # create app interface
        self.setup_interface()
        
        self.display_image()
        
        
    def setup_interface(self):
    
        # create main window
        self.main_window = Gtk.Window(title="Julia Fractals")
        self.main_window.set_border_width(10)
        
        self.main_window.connect("delete-event", Gtk.main_quit)
        
        
        # setup header bar
        self.setup_header_bar()
        
        box = Gtk.Box(orientation='horizontal', spacing=10)
        self.main_window.add(box)
        
        sep = Gtk.Separator(orientation='vertical')
        box.add(sep)
               
        # setup left panel -- container with image parameters
        self.left_box = Gtk.Box(orientation='vertical', spacing=10)
        self.setup_left_box()
        box.add(self.left_box)
        
        sep = Gtk.Separator(orientation='vertical')
        box.add(sep)
        
        # setup right panel -- container with image parameters
        self.right_box = Gtk.Box(orientation='vertical', spacing=10)

        self.setup_right_box()
        box.add(self.right_box)
        
        for name, entry in self.entries.items():
            # copy current values to the defaults
            setattr(self, name + "_default", getattr(self, name))
            self.set_entry_value(entry)
            
        
        sep = Gtk.Separator(orientation='vertical')
        box.add(sep)
        
        # setup image panel -- container with image output
        self.image_box = Gtk.Box(orientation='vertical')
        self.setup_image_box()
        box.add(self.image_box)
        
    
    def setup_header_bar(self):
        '''
        '''
        self.hb = Gtk.HeaderBar()
        self.hb.set_show_close_button(True)
        self.hb.props.title = "Julia Fractal"
        self.main_window.set_titlebar(self.hb)
        
        self.button_save = Gtk.Button(label='Save')
        self.button_save.connect("clicked", self.on_button_save_clicked)
        self.hb.pack_end(self.button_save)        
        
        
    def setup_left_box(self):
        
        # box for 
        box = Gtk.Box(orientation='vertical', spacing=6)
        self.left_box.pack_start(box, False, False, 0)
        
        sep = Gtk.Separator(orientation='horizontal')
        box.add(sep)
        
        self.namecombo = Gtk.ComboBoxText()
        self.namecombo.connect("changed", self.on_namecombo_changed)
        
        for item in ['Julia', 'Mandelbrot']:
            self.namecombo.append_text(item)
        self.namecombo.set_active(0)
        
        box.pack_start(self.namecombo, True, True, 0)
        
        label = Gtk.Label("z = z**n + C", halign=Gtk.Align.START)
        box.pack_start(label, True, True, 0)
        
        label = Gtk.Label("C = x + yi", halign=Gtk.Align.START)
        box.pack_start(label, True, True, 0)
        
        names = ['n', 'x', 'y', 'zmax', 'niter']
        
        entries = {name: self.create_labeled_entry(name, box, 
                              orientation='vertical', spacing=5, xpad=8) 
                   for name in names}
        self.entries.update(entries)                   
        
        sep = Gtk.Separator(orientation='horizontal')
        box.add(sep)
        
        # apply rotation
        button_apply = Gtk.Button(label='Apply')
        button_apply.connect("clicked", self.on_button_apply_clicked)
        self.left_box.pack_start(button_apply, False, False, 0)
        
        # reset rotation
        button_reset = Gtk.Button(label='Reset')
        button_reset.connect("clicked", self.on_button_reset_clicked)
        self.left_box.pack_start(button_reset, False, False, 0)
        
        
    def setup_right_box(self):
    
        # box for 
        box = Gtk.Box(orientation='vertical', spacing=10)
        self.right_box.pack_start(box, False, False, 0)
        
        sep = Gtk.Separator(orientation='horizontal')
        box.add(sep)
        
        names = ['xmin', 'xmax', 'ymin', 'ymax', 'xsize', 'ysize']
        entries = {name: self.create_labeled_entry(name, box, 
                              orientation='horizontal') 
                   for name in names}
        self.entries.update(entries)    
        
        sep = Gtk.Separator(orientation='horizontal')
        box.add(sep)

        label = Gtk.Label("Colormap", halign=Gtk.Align.START)
        box.pack_start(label, True, True, 0)
        
        cmapstore = Gtk.ListStore(str, GdkPixbuf.Pixbuf)
        cmaps = sorted(pl.cm.datad)
        for item in cmaps:
            cmapstore.append([item, None])
            
        self.cmapcombo = Gtk.ComboBox.new_with_model(cmapstore)
        self.cmapcombo.connect("changed", self.on_cmapcombo_changed)
        self.cmapcombo.set_active(0)
        
        renderer = Gtk.CellRendererText()
        self.cmapcombo.pack_start(renderer, True)
        self.cmapcombo.add_attribute(renderer, "text", 0)

        renderer = Gtk.CellRendererPixbuf()
        self.cmapcombo.pack_start(renderer, False)
        self.cmapcombo.add_attribute(renderer, "pixbuf", 1)
        
        box.pack_start(self.cmapcombo, True, True, 0)

        
    def setup_image_box(self):
        
        xx = self.xsize / float(self.dpi)    # inches
        yy = self.ysize / float(self.dpi)    # inches
        
        # TODO: make it resizable
        self.fig = Figure(figsize=(xx, yy), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig) # a gtk.DrawingArea 
        
        # setup drawing canvas
        self.canvas.set_size_request(self.xsize, self.ysize) 
        self.canvas.connect('button-press-event'  , self.on_canvas_button_press)
        self.canvas.connect('button-release-event', self.on_canvas_button_release)
        self.canvas.connect('motion-notify-event' , self.on_canvas_motion_notify)
        self.image_box.add(self.canvas)  
        

    def create_labeled_entry(self, name, parent, orientation='horizontal', spacing=10, xpad=0):
    
        box = Gtk.Box(orientation=orientation, spacing=spacing)
        parent.pack_start(box, True, True, 0)
        
        label = Gtk.Label(label=name, halign=Gtk.Align.START, xpad=xpad)
        box.pack_start(label, True, False, 0)
    
        entry = Gtk.Entry(name=name)
        box.pack_start(entry, True, True, 0)
        
        return entry
        
        
    def set_entry_value(self, entry):
        
        name  = entry.get_name()
        value = getattr(self, name)
        
        if int(value) == value:
            value_str = "{0}".format(value)
        else:
            value_str = "{0:.{1}f}".format(value, self.digits).rstrip("0")
        
        entry.set_text(value_str)
        
        
    def get_entry_value(self, entry, dtype):
        
        text = entry.get_text()
        try:
            val = float(text)
            if dtype == int:
                val = int(val)
        except ValueError:
            val = None
        
        return val


    def on_button_save_clicked(self, widget):
        pass       
        
    
    def on_cmapcombo_changed(self, widget):     
        
        tree_iter = widget.get_active_iter()
        if tree_iter != None:
            model = widget.get_model()
            name, image = model[tree_iter]
            
            self.cmap = name
            if hasattr(self, 'fig'):
                self.display_image()
                self.canvas.draw_idle()
       
        
    def on_namecombo_changed(self, widget):
        
        text = widget.get_active_text()
        print(text)
        if text == 'Mandelbrot':
            self.x = 0
            self.y = 0
            self.set_entry_value(self.entries['x'])
            self.set_entry_value(self.entries['y'])
        
        if hasattr(self, 'fig'):
            self.update_image()
        
        
    def on_button_apply_clicked(self, widget):  
    
        self.update_image()
        
    def on_button_reset_clicked(self, widget):  
        
        for name, entry in self.entries.items():
            setattr(self, name, getattr(self, name + "_default"))
            self.set_entry_value(entry)
    
    
    
    def on_canvas_button_release(self, widget, event):
    
        mapping = {1: 0.75, 3: 1.5}
        if event.button in mapping:
        
            if (self.posx1, self.posy1) == (self.posx2, self.posy2):
                
                factor = mapping[event.button]
                xc = np.interp(event.x, [0, self.xsize-1], [self.xmin, self.xmax])
                yc = np.interp(event.y, [0, self.ysize-1], [self.ymin, self.ymax])
                
                xlen = factor * (self.xmax - self.xmin) 
                ylen = factor * (self.ymax - self.ymin)
                
                self.xmin = xc - xlen/2.0
                self.xmax = xc + xlen/2.0
                
                self.ymin = yc - ylen/2.0
                self.ymax = yc + ylen/2.0
            
            else:
            
                self.xmin = self.posx1
                self.xmax = self.posx2
                
                self.ymin = self.posy1
                self.ymax = self.posy2

            
            for entry in self.entries.values():
                self.set_entry_value(entry)
    
        self.update_image()
        
    
    def on_canvas_motion_notify(self, widget, event):
    
        self.posx2 = np.interp(event.x, [0, self.xsize-1], [self.xmin, self.xmax])
        self.posy2 = np.interp(event.y, [0, self.ysize-1], [self.ymin, self.ymax])

        if event.state & Gdk.EventMask.BUTTON_PRESS_MASK:
            
            if event.state & Gdk.ModifierType.CONTROL_MASK: 
                self.posy2 = self.posy1 + self.posx2 - self.posx1  
            
            if hasattr(self, 'rectangle1'):
                if self.rectangle1 in self.fig.gca().patches:
                    self.rectangle1.remove()
                    self.rectangle2.remove()
            
            self.rectangle1 = Rectangle((self.posx2, (self.ymax + self.ymin) - self.posy2), 
                                        (self.posx1 - self.posx2), -(self.posy1 - self.posy2), 
                                        fill=False, edgecolor='white', linewidth=0.5)
            self.rectangle2 = Rectangle((self.posx2, (self.ymax + self.ymin) - self.posy2), 
                                        (self.posx1 - self.posx2), -(self.posy1 - self.posy2), 
                                        fill=False, edgecolor='black', linestyle='dotted', linewidth=0.5)
            ax = self.fig.gca()                                   
            ax.add_patch(self.rectangle1)
            ax.add_patch(self.rectangle2)
            
            self.canvas.draw_idle()
    
        
    
    def on_canvas_button_press(self, widget, event):
    
        self.posx1 = np.interp(event.x, [0, self.xsize-1], [self.xmin, self.xmax])
        self.posy1 = np.interp(event.y, [0, self.ysize-1], [self.ymin, self.ymax]) 
        
        self.posx2 = self.posx1
        self.posy2 = self.posy1
        
        #self.posx = self.transform_x(event.x)
        #self.posy = self.transform_y(event.y)
            
    
    def run(self):
    
        self.main_window.show_all()
        Gtk.main()
        
    def transform_x(self, xval):
    
        return two_point_interp(xval, 0, self.xsize-1, self.xmin, self.xmin) 
        
    def transform_y(self, yval):
    
        return two_point_interp(yval, 0, self.ysize-1, self.ymin, self.ymin)         
        

    def compute_image(self):
    
        C    = complex(self.x, self.y) 
        xlim = (self.xmin, self.xmax)
        ylim = (self.ymin, self.ymax)
        zz = complex_grid(xlim, ylim, self.xsize, self.ysize)
        
        text = self.namecombo.get_active_text()
        if text == 'Julia':
            return fractal(zz, C, self.n, self.zmax, self.niter)         
        elif text == 'Mandelbrot':
            return fractal(C, zz, self.n, self.zmax, self.niter)    
                
    def display_image(self):
    
        # plot and save the image
        img = self.compute_image()
        
        # clear previous figure
        self.fig.clf()
        # setup plot 
        ax = Axes(self.fig, [0, 0, 1, 1]) # remove outer border  
        ax.set_axis_off()                 # disable axis
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(img, cmap=pl.get_cmap(self.cmap), interpolation='nearest', 
                  extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                  origin='upper', aspect=1.0)
                  
        self.fig.add_axes(ax)

    
    def update_image(self):
    
        for name, entry in self.entries.items():
            
            dtype = int if name in ['n', 'niter', 'xsize', 'ysize'] else float
            val   = self.get_entry_value(entry, dtype)
            if val is not None:
                setattr(self, name, val)
        
        for entry in self.entries.values():
            self.set_entry_value(entry)
        
        self.display_image()
        self.canvas.draw_idle()
    


if __name__ == '__main__':

    app = App()
    app.run()


