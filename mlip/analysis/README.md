Calculate strain (e), off-center displacement (u), and interlayer distance (id) using calc_properties.py  

Usage:  
calc_properties.py {angle}  
Provided is rigid (_rigid) and md averaged (_final) structure files for angles 5 and 20 

Output:  
* {angle}deg_interlayer.csv  
    * columns: x,y,id  
* {angle}deg_strain.csv  
    * columns: layer_number,x,y,z,e_xx,e_xy,e_yx,e_yy,u_x,u_y  
    * Shear strain is calculated as 1/2*(e_xy+e_yx)