#[derive(Debug)]
struct Person<'a> {
    name: &'a str,
    age: u8,
}

// A unit struct
struct Nil;

// A tuple struct
struct Pair(i32, f32);

// A struct with two fields
struct Point {
    x: f32,
    y: f32,
}

// Structs can be reused as fields of another struct
#[allow(dead_code)]
struct Rectangle {
    p1: Point,
    p2: Point,
}

fn rect_area(recta: &Rectangle) -> f32 {
    let Rectangle { p1: Point { x:x1, y:y1}, p2: Point {x:x2, y:y2 } } = *recta;
    (x1 - x2) * (y1 - y2)
}


fn return_square(point: Point, len: f32) -> Rectangle {
    let x1: f32 = point.x + len;
    let y1: f32 = point.y + len;
    let new_point: Point = Point { x: x1, y: y1 };
    let new_rect = Rectangle { p1: point, p2: new_point };

    new_rect
} 
fn main() {
    // Create struct with field init shorthand
    let name = "Peter";
    let age = 27;
    let peter = Person { name, age };
    
    // Print debug struct
    println!("{:?}", peter);
    
    
    // Instantiate a `Point`
    let point: Point = Point { x: 0.3, y: 0.4 };

    // Access the fields of the point
    println!("point coordinates: ({}, {})", point.x, point.y);

    // Destructure the point using a `let` binding
    let Point { x: my_x, y: my_y } = point;

    let _rectangle = Rectangle {
        // struct instantiation is an expression too
        p1: Point { x: my_y, y: my_x },
        p2: point,
    };

    // Instantiate a unit struct
    let _nil = Nil;

    // Instantiate a tuple struct
    let pair = Pair(1, 0.1);

    // Access the fields of a tuple struct
    println!("pair contains {:?} and {:?}", pair.0, pair.1);

    // Destructure a tuple struct
    let Pair(integer, decimal) = pair;

    println!("pair contains {:?} and {:?}", integer, decimal);

    println!("Rectangle Area: {}", rect_area(&_rectangle));
    let point: Point = Point { x:0.3, y:0.4};
    let square: Rectangle = return_square(point, 2.0);

    println!("Area of the Square will be: {}", rect_area(&square));
}

