import sys
from colorterm import *

def health_nb(i, j, grid):
	'''
	returns the number of healthy neighbors of a cell
	in the grid.
	'''

	live_nb = 0
	for a in range(-1, 2):
		testx = (i+a) % len(grid[0])
        for b in range(-1, 2):
            testy = (j+b) % len(grid)
            if b == 0 and a == 0:
                continue
            if grid[testy][testx] == 1:
                live_nb += 1
	return live_nb



def next_step(grid, new_grid):
	'''
	Computes the grid's next step and stores it in new_grid.
	'''
	for i in range(0, len(grid[0])):
		for j in range(0, len(grid)):
			live_nb = health_nb(i, j, grid)
			if grid[j][i]:
				if live_nb < 2 or live_nb > 3:
					new_grid[j][i] = 0
				else:
					new_grid[j][i] = grid[y][x]
			else:
				if live_nb == 3:
					new_grid[j][i] = 1


def read_initial_conf(grid):
    '''
	to initialize the grid with given coordinates by the user
	'''
    done = False
    prompt = 'CONFIG %d: Type coordinates to toggle (or start to finish) %s: '
    config_step = 0
    last_coord = ''
    while True:
        # While input isn't valid, try reading and parsing it
        coord = []
        sys.stdout.write(prompt % (config_step, last_coord))
        while coord == []:
            # Read user's command
            cmd = raw_input()
            # Break if user is finished
            if cmd == 'start' or cmd == '':
                done = True
                break
            try:
                cmd = cmd.split()
                coord = [int(cmd[i]) for i in range(2)]
            except:
                sys.stdout.write(bcolors.RED + '[Invalid input] %s' %
                                 (prompt % (config_step, last_coord))
                                 + bcolors.ENDC)
            last_coord = str(coord)
            config_step += 1
        # Total break if user is finished
        if done:
            break
        # Update grid (it actually toggles the grid position provided)
        grid [coord[1]][coord[0]] = (grid[coord[1]][coord[0]] + 1) % 2
        update_screen(grid)


def update_screen(grid):
	'''
	Update the values in the terminal from updated grid
	'''
	clear_terminal()
	print bcolors.RED + ' GAME OF LIFE' + bcolors.ENDC
	print bcolors.YELLOW + '-'*( len(grid[0]) + 5) + bcolors.ENDC
	print
	for i, line in enumerate(grid):
		print bcolors.BLUE + '%3d ' % i + bcolors.ENDC,
		for element in line:
			if element:
				sys.stdout.write(bcolors.RED + str(element) + bcolors.ENDC)
			else:
				sys.stdout.write('0')
			print
		print bcolors.YELLOW + '-' * ( len(grid[0]) + 5 ) + bcolors.ENDC


def main():
    (width, height) = get_term_size()
    grid = [(width-5)*[0] for i in range(height-5)]


    # Draw initial grid
    update_screen(grid)

    # Read initial config
    read_initial_conf(grid)

    # Step through grid
    prompt = ('ITER %d: Type anything to continue, the number of steps to ' +
              'perform (or quit to exit): ')
    iter_step = 1
    update_screen(grid)
    while True:
        # Wait for user
        play = raw_input('%s' % (prompt % iter_step))
        if play == 'quit':
            break
        try:
            batch_steps = int(play)
        except:
            batch_steps = 1
            pass
        for i in range(batch_steps):
            # Define auxiliary grid matrix
            new_grid = [len(grid[0])*[0] for i in range(len(grid))]
            # Update grid
            next_step(grid, new_grid)
            grid, new_grid = new_grid, grid
            # Print updated grid
            update_screen(grid)
        iter_step += batch_steps



if __name__ == '__main__':
	main()
