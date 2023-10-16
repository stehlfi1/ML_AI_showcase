import npuzzle
 
def is_solvable(env):
   '''
    0) decide size
    1) make pole
    2) count inversions
    2B)check row inversions

    PS) Proč solvable nemůže dostavat size paramentr trošku nechapu, nojono ted kod bude ugly
   '''
   size = 4
   a = [0] * 16
   sizedicider = 0
   inverze = 0  
   for row in range(size):
        for col in range(size):
            try:
                a[sizedicider]  = env.read_tile(row, col)
                sizedicider = sizedicider + 1
            except:
                continue                 
   for i in range(sizedicider):
      for j in range(i+1, sizedicider):
         if(a[j] is None):
            continue
         if(a[i] is None):
            blankspot = i + 1
            continue
         if(a[i] > a[j]):
            inverze = inverze + 1
   #print("inverze:")
   #print(inverze)
   if(sizedicider == 16):
      if((blankspot % 8)  < 5):
          inverze = inverze + 1
   if(inverze % 2 == 0):
      return True
   else:
      return False 

if __name__=="__main__":
   env = npuzzle.NPuzzle(4) # env = npuzzle.NPuzzle(4)
   env.reset()
   env.visualise()
   # just check
   print(is_solvable(env))