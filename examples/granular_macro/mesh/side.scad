rotate(a=[90,0,0]){
    translate([-405,-405,-3]){
        linear_extrude(height =3, $fn = 209){
        square([810,810],True, $fn = 20);
    }
};

    translate([-405,-405, 100]){
        linear_extrude(height = 3, $fn = 290){
        square([810,810],True,$fn = 20);
    }
};
}