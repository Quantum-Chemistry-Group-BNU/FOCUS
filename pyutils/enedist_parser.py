import subprocess

def get_file_tail(filename, n=10):
    """获取文件最后n行，推荐使用"""
    try:
        # 方法1：使用tail命令（最快）
        result = subprocess.run(['tail', '-n', str(n), filename],
                              capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        # 备用方案：纯Python实现
        try:
            from collections import deque
            with open(filename, 'r') as f:
                return list(deque(f, maxlen=n))
        except Exception:
            return []  # 或者抛出异常

def parser(nmax, dirname = "results", debug=False):
   print('\n[parser] nmax=',nmax)
   rlst = []
   ilst = []
   vlst = []
   status = []
   for i in range(nmax):
      fname = dirname + '/output.enedist_' + str(i)
      # 使用
      lines = get_file_tail(fname, 10)
      finish = False
      for line in lines:
         if ">>> Final enedist: (omegaR,omegaI)" in line:
            if debug: print(line)
            s = line.split()
            rlst.append(s[4])
            ilst.append(s[5])
            vlst.append(s[-1])
            finish = True 
      print('processing file=',fname,' exist=',len(lines)>0,' finish=',finish)
      status.append(finish)

   print('final results:')
   for i in range(len(rlst)):
      print(rlst[i],ilst[i],vlst[i])
   nfinish = sum(item is True for item in status)
   print('\nno. of finished jobs=',nfinish,' nmax=',nmax)
   return 0

if __name__ == "__main__":
   
    import sys

    if len(sys.argv) > 1:
        nmax = int(sys.argv[1])
        dirname = sys.argv[2] if len(sys.argv) > 2 else "results"
    else:
       print('error: nmax needs to be defined') 
       exit(1)

    parser(nmax, dirname)
