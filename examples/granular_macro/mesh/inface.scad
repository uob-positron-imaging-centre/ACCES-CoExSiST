rotate(a=[90,0,0]){
translate([-200,200,-0]){#
    rotate(a = [90,0,0]){
        linear_extrude(height =10,$fn = 20){
            
                square([400,100],True);
            }
    }
};
}