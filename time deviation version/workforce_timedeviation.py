from ast import increment_lineno
from sqlite3 import Row
import sys
import docplex.mp
from collections import namedtuple
import pandas as pd
import numpy as np
import math

skillsTable=pd.read_csv('skills1.csv',sep=',',header=0)
eskillsTable = pd.read_csv('Employee Skillset1.csv', sep=",", header=0)
avilibilityTable = pd.read_csv('Employee Availablity1.csv', sep=",", header=0)
tasksTable = pd.read_csv('Production Needs1.csv', sep=",", header=0)
o_tasklist =[tasksTable["Need ID"][i] for i in range(len(tasksTable))]
timelist=[x/2 for x in list(range(48))]
def myround(x):
  if x==0:
    return 0
  else:
    return int(round(x)+1)

def EmployeeTime(x):
  x.insert(1,"Time","")
  for i in range(len(x)):
    start_str=x.iloc[i,3]
    end_str=x.iloc[i,4]
    lunchs_str=x.iloc[i,5]
    lunche_str=x.iloc[i,6]
    starttime=int(start_str[:len(start_str)-3])*2+myround(int(start_str[-2:])/60)
    lunchs=int(lunchs_str[:len(lunchs_str)-3])*2+myround(int(lunchs_str[-2:])/60)
    lunche=int(lunche_str[:len(lunche_str)-3])*2+myround(int(lunche_str[-2:])/60)
    endtime=int(end_str[:len(end_str)-3])*2+myround(int(end_str[-2:])/60)
    x.iat[i,1]=timelist[starttime:lunchs]+timelist[lunche:endtime]
  employee_time={}
  TeTime = namedtuple("TeTime", ["Employee", "timeset"])
  for et in x.iloc[:,0:2].itertuples(index=False):
    ett=TeTime(*et)
    employee_time.setdefault(ett.Employee,ett.timeset)
  return(employee_time) 
employee_time=EmployeeTime(avilibilityTable)

tasksTable.insert(1,"Time","")
nb_task=len(tasksTable)
for i in range(nb_task):
    taskstart_str=tasksTable.iloc[i,4]
    taskend_str=tasksTable.iloc[i,6]
    minus=tasksTable.iloc[i,7]
    plus=tasksTable.iloc[i,8]
    taskstart=int(taskstart_str[:len(taskstart_str)-3])*2+myround(int(taskstart_str[-2:])/60)
    taskend=int(taskend_str[:len(taskend_str)-3])*2+myround(int(taskend_str[-2:])/60)
    minus_num=int(minus[:len(minus)-3])*2+myround(int(minus[-2:])/60)
    plus_num=int(plus[:len(plus)-3])*2+myround(int(plus[-2:])/60)
    tasksTable.iat[i,1]=timelist[taskstart:taskend]
    for j in range(1,minus_num+1):
      x_new_row=pd.DataFrame({'Need ID':[(tasksTable.iloc[i,0],"-",j)],'SkillsetID1':[tasksTable.iloc[i,2]],'Time':[timelist[taskstart-j:taskend-j]]})
      tasksTable=pd.concat([tasksTable,x_new_row],ignore_index=True)
    for j in range(1,plus_num+1):
      x_new_row=pd.DataFrame({'Need ID':[(tasksTable.iloc[i,0],"+",j)],'SkillsetID1':[tasksTable.iloc[i,2]],'Time':[timelist[taskstart+j:taskend+j]]})
      tasksTable=pd.concat([tasksTable,x_new_row],ignore_index=True)
    tasksTable.iat[i,0]=(tasksTable.iloc[i,0],"0")
#o_tasklist = [tasksTable["Need ID"][i] for i in range(nb_task)]
task_time={}
TtTime = namedtuple("TeTime", ["Task", "timeset"])
for tt in tasksTable.iloc[:,0:2].itertuples(index=False):
    ttt=TtTime(*tt)
    task_time.setdefault(ttt.Task,ttt.timeset)

def TaskSkill(x):
  TtSkill = namedtuple("TtSkill", ["Task", "skillset"])
  task_skill={}
  for tsk in x.iloc[:,[0,2]].itertuples(index=False):
    tskt=TtSkill(*tsk)
    task_skill.setdefault(tskt.Task,tskt.skillset)
  return(task_skill)
task_skill=TaskSkill(tasksTable)
def EmployeeSkill(x):
  TeSkill = namedtuple("TeSkill", ["Employee", "skillset"])
  employee_skills = {}
  for esk in x.itertuples(index=False):
    eskt= TeSkill(*esk)
    employee_skills.setdefault(eskt.Employee, []).append(eskt.skillset)
  return(employee_skills)
employee_skills=EmployeeSkill(eskillsTable)

skillslist = [skillsTable["name"][i] for i in range(len(skillsTable))]
employeelist = [avilibilityTable["Employee ID"][i] for i in range(len(avilibilityTable))]
tasklist = [tasksTable["Need ID"][i] for i in range(len(tasksTable))]





##Import the model
from docplex.mp.model import Model
mdl = Model("employee")

##Define the decision variables
employee_task_vars = mdl.binary_var_matrix(employeelist, tasklist, 'EmployeeAssigned')
total_number_of_assignments = mdl.sum(employee_task_vars[e,t] for e in employeelist for t in tasklist)

employee_work_time_vars = mdl.continuous_var_dict(employeelist, lb=0, name='EmployeeWorkTime')
work_start_var=mdl.continuous_var_dict(employeelist)
work_end_var=mdl.continuous_var_dict(employeelist)
for e in employeelist:
    mdl.add_constraint(work_end_var[e]==mdl.max(employee_task_vars[e,t]*task_time[t][-1] for t in tasklist))
    mdl.add_constraint(work_start_var[e]==mdl.max(employee_task_vars[e,t]/task_time[t][0] for t in tasklist))
    mdl.add_constraint(employee_work_time_vars[e]== work_end_var[e]+work_start_var[e])
total_work=mdl.sum(employee_work_time_vars[e] for e in employeelist)

##Add constrains
#constrain 1:when employee are unavaibele, he can not assigned any tasks starting that time.
for e in employeelist:
  for t in tasklist:
    if all(elem in employee_time[e] for elem in task_time[t])==False:
      mdl.add_constraint(employee_task_vars[e, t] == 0)
#constrain 2:can not assign overlap task; employee perform one task at each t 
nb_tasks = len(tasksTable)
for i1 in range(nb_tasks):
    for i2 in range(i1 + 1, nb_tasks):
      s1=tasksTable.iloc[i1,0]
      s2=tasksTable.iloc[i2,0]
      if len(set(tasksTable.iloc[i1,1]) & set(tasksTable.iloc[i2,1]))>0:
        for n in employeelist:
          mdl.add_constraint(employee_task_vars[n, s1] + employee_task_vars[n, s2] <= 1)
#constrain3: enforce skill requirements for selected task.
for task in tasklist:
  for employee in employeelist:
    if len(set(employee_skills[employee])& set(task_skill[task]))==0:#check if two lists have common elements
      mdl.add_constraint(employee_task_vars[employee, task] == 0)
#constrain4: one task only need one employee
for s in tasklist:
    total_assigned= mdl.sum(employee_task_vars[n, s] for n in employeelist)
    mdl.add_constraint(total_assigned<= 1)
#constrain5: only pick one time window for each task
#mdl.add constraint()
number_tasks= mdl.sum(employee_task_vars[e,t] for e in employeelist)
nb_tasks = len(tasksTable)
for i1 in range(nb_tasks):
    s1=tasksTable.iloc[i1,0]
    total_assigned_s1= mdl.sum(employee_task_vars[n, s1] for n in employeelist)
    for i2 in range(i1 + 1, nb_tasks):     
      s2=tasksTable.iloc[i2,0]
      total_assigned_s2= mdl.sum(employee_task_vars[n, s2] for n in employeelist)
      if s1[0]==s2[0]:
          mdl.add_constraint(total_assigned_s1 + total_assigned_s2<= 1)

##informations of model
mdl.add_kpi(total_number_of_assignments,"number of assigned task")
mdl.print_information()

##Aim function and slove
mdl.maximize(10*total_number_of_assignments-0.1*total_work)
s = mdl.solve()

##solution and total work hour
mdl.report()
mdl.print_solution()
work_time=[]
for e in employeelist:
  start_time= 24
  end_time=0
  for t in tasklist:
    if employee_task_vars[e,t].solution_value >0:
      start_time = min(task_time[t][0],start_time)
      end_time = max(task_time[t][-1],end_time)
  work_time.append(max(end_time-start_time+0.5,0))
  #print(work_time)
total_work_hours=sum(work_time)
print("The total work hours is",total_work_hours,"hours")

def unassignTask(assignlist,o_tasklist):
  for e in employeelist:
    for t in tasklist:
      if employee_task_vars[e,t].solution_value > 0:
        if (t[0] in o_tasklist):
          assignlist.append(t[0])
  unassignlist=list(set(o_tasklist)-set(assignlist))
  print("Unassigned Task:",unassignlist)
  for i in unassignlist:
    print(i,": from",task_time[(i,"0")][0],"to",task_time[(i,"0")][-1]+0.5,",required skillset:",task_skill[(i,"0")])
unassignTask([],o_tasklist)




###draw the solution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

##create label
def createLabels(ax, title, xlabel, ylabel):
  tmin=int(min(employee_time.values())[0])
  tmax=math.ceil(int(max(employee_time.values())[-1])+0.5)
  plt.xticks(np.arange(tmin,tmax,0.5),["{}".format(str(s)) for s in timelist[tmin:tmax]])
  plt.xlim([tmin,tmax])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid()
  ax.set_title(title)

colorBySkills = {}
for d, c in zip(skillslist, ['r', 'm', 'b', 'g', 'y', 'c']):
    colorBySkills[d] = c
colors=['k','r', 'm', 'b', 'g', 'y', 'c']
labels=['avilibility','A','B','C','D','E','F']
red_patch = [mpatches.Patch(color=colors[i], label=labels[i])for i in range(len(labels))]


def displayTasksAssignmentsGantt(ax):
    ylabels, tickloc = [], []
    for i in range(len(employeelist)):
      n=employeelist[i]
      lunchs_str=avilibilityTable["Lunch Start"][i]
      lunchs=(int(lunchs_str[:len(lunchs_str)-3])*2+myround(int(lunchs_str[-2:])/60))/2
      ax.bar(employee_time[n][0],0.3,width=lunchs-employee_time[n][0],bottom=i+0.5,color='k',align='edge')
      ax.bar(lunchs+1,0.3,employee_time[n][-1]-lunchs-0.5,bottom=i+0.5,color='k',align='edge')
      for s in tasklist:
        if employee_task_vars[n,s].solution_value > 0:
          ax.bar(task_time[s][0], 0.4,
                 width=task_time[s][-1] - task_time[s][0]+0.455, bottom=i + 0.1,
                 color=colorBySkills[task_skill[s]],align='edge')
                  
      ylabels.append("{}:{}".format(str(n),employee_skills[n])) # (3)number of total hours works
      tickloc.append(i + 0.5)

    plt.ylim(0, len(employeelist))
    plt.yticks(tickloc, ylabels)
    plt.legend(handles=[red_patch][0]) #[0] beacuse error:'list' object has no attribute 'get_label'
    plt.show()
    # Create labels on x/y axis
    createLabels(ax, 'Taks Assignments', 'Hours Of The Day', 'Employee ID & Skill')

fig = plt.figure(figsize=[14,12])
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax = plt.subplot(gs[0])
displayTasksAssignmentsGantt(ax)

