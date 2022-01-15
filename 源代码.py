# coding=utf8
import random
import time
import tkinter as tk
import pandas as pd
import numpy as np
from tkinter.simpledialog import *
from tkinter import *

GREEDY=0.9
ALPHA=0.1
GAMMA=0.9
sum=0

# random maze creater from github
def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 1:
                        return True
                    if res[r_new][c_new] != -1:
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice([0, -1], (size, size), p=[p, 1 - p])
        res[0][0] = 0
        res[-1][-1] = 1
        valid = is_valid(res)
    return res

# INITIAL=[0, -1, 0, 0, 0, 0, 0, 0,
#         0, -1, 0, 0, -1, -1, 0, 0,
#         0, -1, 0, -1, 0, 0, -1, 0,
#         0, -1, 0, -1, 0, 0, -1, 0,
#         0, -1, 0, -1, 0, 0, 0, 0,
#         0, -1, 0, -1, 0, -1, -1, -1,
#         0, -1, 0, -1, 0, 0, -1, 0,
#         0, 0, 0, -1, 0, 0, 0, 1]

INITIAL=generate_random_map()

class Maze(tk.Tk):
    PIXEL = 90
    def generate_random_map(size=8, p=0.8):
        """Generates a random valid map (one that has a path from start to goal)
        :param size: size of each side of the grid
        :param p: probability that a tile is frozen
        """
        valid = False

        # DFS to check that it's a valid path.
        def is_valid(res):
            frontier, discovered = [], set()
            frontier.append((0, 0))
            while frontier:
                r, c = frontier.pop()
                if not (r, c) in discovered:
                    discovered.add((r, c))
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for x, y in directions:
                        r_new = r + x
                        c_new = c + y
                        if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                            continue
                        if res[r_new][c_new] == 1:
                            return True
                        if res[r_new][c_new] != -1:
                            frontier.append((r_new, c_new))
            return False

        while not valid:
            p = min(1, p)
            res = np.random.choice([0, -1], (size, size), p=[p, 1 - p])
            res[0][0] = 0
            res[-1][-1] = 1
            valid = is_valid(res)
        return res
    
    global INITIAL

    def printini(self):
        for c in range(0, 8):
            print(INITIAL[c])

    def xz(self):
        answer=tk.messagebox.askokcancel('本轮学习完毕','您是否要继续训练？')
        return answer

    #GUI env from Github

    def __init__(self):
        super().__init__()
        self.title('Maze')
        h = 8 * self.PIXEL
        w = 8 * self.PIXEL
        self.geometry('{0}x{1}'.format(h+10, w+10))
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        for c in range(1, 8):
            self.canvas.create_line(c * self.PIXEL, 0, c * self.PIXEL, h)
        for r in range(1, 8):
            self.canvas.create_line(0, r * self.PIXEL, w, r * self.PIXEL)
        self._draw_rect(0, 0, 'yellow')
        # INITIAL=self.generate_random_map()
        for a in range (0,8):
            for b in range(0,8):
                if(INITIAL[b][a]==-1):
                    self._draw_rect(a,b,'black')
        self._draw_rect(7, 7, 'red')
        self.rect = self._draw_oval(0, 0, 'green')
        self.canvas.pack()
        
    def click(self):
        return 1
    
    def give_message(self):
        tk.messagebox.showinfo(title='参数选择完成', message=('您选择的参数为：',"GREEDY=",GREEDY,'ALPHA=',ALPHA,'GAMMA=',GAMMA,'接下来将对模型进行首轮50次训练'))  

    def ending(self):
        tk.messagebox.showinfo(title='成功', message=('寻路成功，Agent已抵达终点，共行进',sum,'步'))  


    def index_input(self):
        a=askfloat("请输入","test")
        b=askfloat("test","test")
        # s=askstring('请输入','请输入一串文字')
        return [a,b]

    def destroy_winnew(root,a,b,c):
        print(a,b,c)
        global GREEDY
        global ALPHA
        global GAMMA
        GREEDY=a
        ALPHA=b
        GAMMA=c
        

    def newwind(self):
        winNew = Toplevel(self)
        winNew.geometry('360x360')
        winNew.title('参数输入')
        lb2 = Label(winNew,text='\n随机迷宫生成完毕，请输入Q-Learning的参数')
        lb2.pack()
        var1=DoubleVar()
        var2=DoubleVar()
        var3=DoubleVar()
        scl_1=Scale(winNew,orient=HORIZONTAL,length=200,from_=0.0,to=1.0,label='\n请拖动滑块选择xxxxxx取值',tickinterval=0.2,resolution=0.01,variable=var1,command=self.get_1(self.scl_1.get))
        scl_1.pack()
        scl_2=Scale(winNew,orient=HORIZONTAL,length=200,from_=0.0,to=1.0,label='\n请拖动滑块选择xxxxxx取值',tickinterval=0.2,resolution=0.01,variable=var2)
        scl_2.pack()
        scl_3=Scale(winNew,orient=HORIZONTAL,length=200,from_=0.0,to=1.0,label='\n请拖动滑块选择xxxxxx取值',tickinterval=0.2,resolution=0.01,variable=var3)
        scl_3.pack()
        global GREEDY
        global ALPHA
        global GAMMA
        # self.destroy_winnew(scl_1.get(),scl_2.get(),scl_3.get())
        # b = Button(winNew, text="执行", command=).pack()
        # c=Button(winNew, text="执行", command=winNew.quit).pack()
        GREEDY=scl_1.get()
        ALPHA=scl_2.get()
        GAMMA=scl_3.get()
        winNew.mainloop()

    def _draw_rect(self, x, y, color):
        padding = 0 
        coor = [self.PIXEL * x + padding, self.PIXEL * y + padding, self.PIXEL * (x + 1) - padding,
                self.PIXEL * (y + 1) - padding]
        return self.canvas.create_rectangle(*coor, fill=color)

    def _draw_oval(self, x, y, color):
        padding = 0 
        coor = [self.PIXEL * x + padding, self.PIXEL * y + padding, self.PIXEL * (x + 1) - padding,
                self.PIXEL * (y + 1) - padding]
        return self.canvas.create_oval(*coor, fill=color)

    def move_agent_to(self, position):
        coor_old = self.canvas.coords(self.rect)
        x, y = position % 8, position // 8  
        padding = 0  
        coor_new = [self.PIXEL * x + padding, self.PIXEL * y + padding, self.PIXEL * (x + 1) - padding,
                    self.PIXEL * (y + 1) - padding]
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1] 
        self.canvas.move(self.rect, dx_pixels, dy_pixels)
        self.update() 


class Agent(object):
    def __init__(self):
        tmp=[]
        for i in range (0,8):
            tmp.extend(INITIAL[i])
        self.position=0
        self.map = tmp
        self.q_table = pd.DataFrame(data=[[0]*4]*64,index=range(64),columns=['u','d','l','r'])

    def Bellman_Equation(self,position,action,a,b):
        self.q_table.loc[position,action]+=ALPHA*(a+GAMMA*b.max()-self.q_table.loc[position,action])

    def choose(self,position,greedy=GREEDY):
        tmp=random.uniform(0, 1)
        cur_actions=self.exclude(position)
        s=[self.q_table.loc[position,cur_actions[0]]]
        if (tmp>=greedy):
            action = random.choice(cur_actions)
        else:
            for c in range(1,len(cur_actions)):
                s.extend([self.q_table.loc[position,cur_actions[c]]])
            s.sort(reverse=True)
            ans=[]
            for c in range(0,len(cur_actions)):
                if(self.q_table.loc[position,cur_actions[c]]==s[0]):
                    ans.extend(cur_actions[c])
            action=random.choice(ans)
        return action

    def move(self, position, action):
        if (action=='u'):
            next_position=position-8
        if (action=='d'):
            next_position=position+8
        if (action=='l'):
            next_position=position-1
        if (action=='r'):
            next_position=position+1
        return next_position

    def exclude(self, position):
        tmp=set('udlr')
        if (position//8==0):
            tmp-={'u'}
        if (position//8==7):
            tmp-={'d'}
        if (position%8==0):
            tmp-={'l'}
        if (position%8==7):
            tmp-={'r'}
        return list(tmp)

    def Q_learning(self,times,env):
        for i in range(0,times):
            print("正在进行第",i+1,"轮训练")
            cur=0
            env.move_agent_to(cur)
            while (cur!=63):
                actions=self.choose(cur,GREEDY)
                next_position=self.move(cur, actions)
                next_q=self.map[next_position]
                # next_table= self.get_q_values(next_position)
                next_table= self.q_table.loc[next_position,self.exclude(next_position)]
                self.Bellman_Equation(cur,actions,next_q,next_table)
                env.move_agent_to(next_position)
                cur=next_position
        print('本轮训练完成')


    def display(self,env):
        global sum
        cur=0
        env.move_agent_to(cur)
        while cur!=63:
            actions=self.choose(cur,1) #贪心拉满，直接选择最优解
            next_position=self.move(cur,actions)
            env._draw_rect(cur%8,cur//8,'green')
            if(actions=='r'):
                print('Right')
            elif(actions=='l'):
                print('Left')
            elif(actions=='u'):
                print('Up')
            else:
                print('Down')
            cur=next_position
            sum=sum+1
            env.move_agent_to(cur)
            time.sleep(0.1)


if __name__ == '__main__':
    # INITIAL=generate_random_map()
    print("Initial maze:")
    for c in range(0,8):
        print(INITIAL[c])
    env = Maze()
    # env.index_input()
    agent = Agent()

    def showNum():
        # print(scale1.get())
        global GREEDY
        global ALPHA
        global GAMMA
        GREEDY=scale1.get()
        ALPHA=scale2.get()
        GAMMA=scale3.get()
        win.destroy()
        env.give_message()
        agent.Q_learning(times=50, env=env)
        while(env.xz()):
            agent.Q_learning(times=50,env=env)
        agent.display(env)
        env.ending()
        env.destroy()
        
    win=tk.Tk()
    win.title("请选择Q-Learning的参数")
    win.geometry("480x360")
    win.wm_attributes('-topmost',1)
    tmp=tk.Label(win,text='\n随机迷宫生成完毕,请选择Q-Learning的参数\n',font=("黑体",15)).pack()
    scale1 = tk.Scale(win,orient=HORIZONTAL,length=300,from_=0.0,to=1.0,label='\n请拖动滑块选择GREEDY取值',font=("黑体",10),tickinterval=0.2,resolution=0.01)
    scale1.pack()
    scale1.set(0.9) 

    scale2 = tk.Scale(win,orient=HORIZONTAL,length=300,from_=0.0,to=1.0,label='\n请拖动滑块选择学习率参数ALAPHA取值',font=("黑体",10),tickinterval=0.2,resolution=0.01)
    scale2.pack()
    scale2.set(0.1)
    
    scale3 = tk.Scale(win,orient=HORIZONTAL,length=300,from_=0.0,to=1.0,label='\n请拖动滑块选择奖励值递减参数GAMMA取值',font=("黑体",10),tickinterval=0.2,resolution=0.01)
    scale3.pack()
    scale3.set(0.9)
    a=tk.Label(win,text="").pack()
    tk.Button(win, text="我选好了",width=10,height=2,font=("黑体",10),command=showNum).pack()
    
    env.mainloop()