 '''
        Method that must be implemented by you. 
        Expects to return a path_section as a list of positions [(x1, y1), (x2, y2), ... ].

        // A* (star) Pathfinding

        // Initialize both open and closed list                                          DONE
        let the openList equal empty list of nodes                                       DONE
        let the closedList equal empty list of nodes                                     DONE
        // Add the start node                                                            DONE
        put the startNode on the openList (leave it's f at zero)                         DONE
        // Loop until you find the end                                                   DONE
        while the openList is not empty                                                  x
            // Get the current node                                                      x
            let the currentNode equal the node with the least f value                    x
            remove the currentNode from the openList                                     x
            add the currentNode to the closedList                                        x
            // Found the goal                                                             DONE
            if currentNode is the goal                                                    DONE
                Congratz! You've found the end! Backtrack to get path                     DONE
            // Generate children                                                         x
            let the children of the currentNode equal the adjacent nodes                 x
            
            for each child in the children                                               x
                // Child is on the closedList                                            x
                if child is in the closedList                                            x
                    continue to beginning of for loop                                    x
                // Create the f, g, and h values                                         x
                child.g = currentNode.g + distance between child and current             x
                child.h = distance from child to end                                     x
                child.f = child.g + child.h                                              x
                // Child is already in openList                                          x
                if child.position is in the openList's nodes positions                   x 
                    if the child.g is higher than the openList node's g                  x
                        continue to beginning of for loop                                x
                // Add the child to the openList                                         x
                add the child to the openList                                            x
        '''
         '''
                if(child_node in open_array):
                    print("    skipping lmao  open", " h: ", child_node.h,"g :", child_node.g, child_node.position) 
                '''
                 '''
                for i in range (len(open_array)):
                    temp = open_array[i]
                    print("open arr:",i, temp.position, temp.f)
                '''
                 #print("    h: ", child_node.h,"g :", child_node.g, child_node.position)
                  #child_node.g = g_function(child_node.position, start_node.position, child_node.cost)
                  #print("    skipping lmao close", " h: ", child_node.h,"g :", child_node.g, child_node.position)

if(goal_reached):
            return path[::-1] # cool trick everybody knows to reverse list
        else:
            return None