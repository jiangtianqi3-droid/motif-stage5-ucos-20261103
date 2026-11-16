using System;
using System.IO;
using System.Drawing;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Web.Script.Serialization;

class ExperimentProgress : Form {
    Label info = new Label();
    ProgressBar bar = new ProgressBar();
    Timer timer = new Timer();
    string status;
    [STAThread] static void Main() {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new ExperimentProgress());
    }
    ExperimentProgress() {
        Text = "B2+D* 实验进度";
        ClientSize = new Size(700, 270);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Microsoft YaHei UI", 10);
        info.SetBounds(20, 20, 660, 180);
        bar.SetBounds(20, 210, 660, 25);
        Controls.Add(info); Controls.Add(bar);
        status = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "status.json"));
        timer.Interval = 2000;
        timer.Tick += delegate { RefreshStatus(); };
        Shown += delegate { RefreshStatus(); Activate(); };
        FormClosed += delegate { timer.Dispose(); };
        timer.Start();
    }
    string Get(Dictionary<string,object> s, string key) {
        object value; return s.TryGetValue(key, out value) && value != null ? value.ToString() : "—";
    }
    void RefreshStatus() {
        try {
            string json;
            using(var stream = new FileStream(status, FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete))
            using(var reader = new StreamReader(stream)) { json = reader.ReadToEnd(); }
            var s = new JavaScriptSerializer().Deserialize<Dictionary<string,object>>(json);
            double done, total, eta;
            Double.TryParse(Get(s,"done"),out done); Double.TryParse(Get(s,"total"),out total);
            Double.TryParse(Get(s,"eta_seconds"),out eta);
            bar.Value = total > 0 ? (int)Math.Max(0,Math.Min(100,100*done/total)) : 0;
            info.Text = "状态："+Get(s,"state")+" / "+Get(s,"stage")+"\r\n"+
                "图谱："+Get(s,"graph")+"；种子："+Get(s,"seed")+"；阶段："+Get(s,"split")+"\r\n"+
                "当前阶段："+done+" / "+total+"（"+bar.Value+"%）\r\n"+
                "本阶段剩余约："+Math.Round(eta)+" 秒；本次进程新增 API："+Get(s,"api_calls")+"\r\n"+
                "输入 Token："+Get(s,"prompt_tokens")+"；输出 Token："+Get(s,"completion_tokens")+"\r\n"+
                "累计估算费用：USD "+Get(s,"estimated_cost_usd_upper")+"\r\n"+
                "更新时间："+Get(s,"updated_at");
            if(Get(s,"state")=="failed") info.Text += "\r\n错误："+Get(s,"error");
        } catch(Exception ex) { info.Text = "等待状态文件刷新："+ex.Message; }
    }
}
