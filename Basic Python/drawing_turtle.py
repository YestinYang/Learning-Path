import turtle

def draw_square(turtle_name):
    for i in range(4):
        turtle_name.forward(100)
        turtle_name.right(90)

def draw_tri(turtle_name, dist):
    for i in range(3):
        turtle_name.forward(dist)
        turtle_name.left(120)

def draw_tri_grid(turtle_name, init_dist):
    for i in range(3):
        # second large triangle
        for j in range(3):
            # third large triangle
            for k in range(3):
                # fill the smallest triangle
                turtle_name.begin_fill()
                draw_tri(turtle_name, init_dist/8)
                turtle_name.end_fill()
                turtle_name.forward(init_dist/4)
                turtle_name.left(120)
            turtle_name.forward(init_dist/2)
            turtle_name.left(120)
        turtle_name.forward(init_dist)
        turtle_name.left(120)
        
def draw_art():
    window = turtle.Screen()
    window.bgcolor("white")
    
    Leo = turtle.Turtle()
    Leo.color("blue")
    Leo.fillcolor("green")
    Leo.shape("turtle")
    Leo.speed(8)

    Raph = turtle.Turtle()
    Raph.color("red")
    Raph.shape("turtle")
    Raph.speed(5)

    # for i in range(50):
    #     draw_square(Leo)
    #     Leo.right(360/50)
    
    # draw_tri(Raph)

    draw_tri_grid(Leo, 200)

    window.exitonclick()

draw_art()