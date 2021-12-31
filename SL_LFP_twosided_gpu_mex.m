function SL_comp=SL_LFP_twosided_gpu_mex(rawdata_valid,system_fs,starts,ends,chans,LP,HS,n_rec,p_ref,stps,delays,fname,n_level,proc_avg)
% revised by zifang zhao @ 2013-10-8 增加输入proc_avg作为判断是否应该进行对trial平均的操作
% renamed from SL_LFP_gpu,改为双侧计算 @ 2012-12-28 by zifang zhao
% revised by zifang zhao @ 2012-12-20 增加输入level,用于噪音筛除电平(abs)
% revised by zifang zhao @ 2012-12-12 修正读取fail后fail文件继续存在的问题
% revised by zifang zhao @ 2012-12-9 增加文件名输入，将temp文件对应到具体计算文件(temp_fail结构)
% revised by zifang zhao @ 2012-12-6 优化代码结构，去掉channels的cell结构
% revised by zifang zhao @ 2012-12-5 修正时移，修正时移坐标输出
% revised by zifang zhao @ 2012-9-12
% 修改了计算时对数据长度的限制，取消了trial的长度，即只根据起始时间和参数进行计算
% revised by zifang zhao @ 2012-9-10 使用新的SL_single_k_v4
% revised by zifang zhao @ 2012-9-9 更换核心函数为9-9新版SL_single_k_v3
% revised by zifang zhao @ 2012-9-5 修正stp_len的时间判断错误（由length()改为max()）
% revised by Zifangzhao @ 2012-6-19 增加fs到SL结构体中
% created by ZifangZhao @ 2012-6-7

if nargin<14
    proc_avg=0;
end

%% 参数变换
sampling_delay=ceil(system_fs./HS/3);% L
embedding=ceil(3*HS./LP);% m
w1=sampling_delay.*(embedding-1);
w2=ceil((n_rec/p_ref+2*w1-1)/2);

% k=parallel.gpu.CUDAKernel('cal_dist_2.ptx','cal_dist_2.cu');%创建核函数
% k.ThreadBlockSize=[1024 1 1];
% k.GridSize=[64 1];

%% 初始化

stp_range=max(stps+1);
delay_range=max(delays+1);
channels_len =  length(chans);
trim_offset=-ceil(w2);
trim_len=ceil(stp_range+2*w2+delay_range);

%%noise ejection
[starts ends]=noise_eject(rawdata_valid,system_fs,starts+trim_offset/1000,ends+trim_offset/1000,trim_len,n_level); %双边谱，start应从-trim_len/2开始
trials       =  length(starts);
if proc_avg==0;
    SL           =  cell(trials,1);
else
    SL           =  cell(1,1);
    data_temp=zeros(length(chans),trim_len,trials);
end

%% 滤波
rawdata_filtered=cellfun(@(x) fft_filter(x,LP,HS,system_fs),rawdata_valid,'UniformOutput',0);

%using FIR1
% [b a]=fir1(4,[LP HS]*2/system_fs);
% rawdata_filtered=cellfun(@(x) filtfilt(b,a,x),rawdata_valid,'UniformOutput',0);

clear('rawdata_valid');
%% check and start ticking
try
    toc;
catch err
    disp(['There is an Error, Cupcake! ' err.message]);
    tic;
end
multiWaitbar('Trial:',0,'color',[0.5,0.6,0.7]);

%% Data processing
time_ind_back=0;

trial_temp_file=dir([fname '_fail.mat']);   %如果存在之前的fail文件，则继续
for trial = 1:trials;
    if ~isempty(trial_temp_file)&&~proc_avg
        try
            load(trial_temp_file(1).name);
            system(['del ' trial_temp_file(1).name(1:end-8) 'temp.mat']); %读取完毕，删除原先的temp
            system(['rename ' trial_temp_file(1).name ' ' trial_temp_file(1).name(1:end-8) 'temp.mat']); %将fail改为temp
        catch err
            system(['del ' trial_temp_file(1).name]); %如果temp文件出错则删除之
        end
    end
    if trial>time_ind_back
        fprintf(2,['Trial ' num2str(trial) ' in processing \n']);
        starts_temp=starts(trial);
        ends_temp=ends(trial);
        idx_start=round(starts_temp.*system_fs)+trim_offset;   
        idx_end=round(ends_temp.*system_fs)+trim_offset;   
        data_len=length(rawdata_filtered{1});
        if idx_start<=0;
            idx_start=1;
        end
        if idx_end<=0;
            idx_end=1;
        end
        if idx_start>data_len
            idx_start=data_len;
        end
        if idx_end>data_len
            idx_end=data_len;
        end
        [ch1 ch2]=meshgrid(1:channels_len,1:channels_len);
        SL{trial}=arrayfun(@(x,y) zeros(length(delays),length(stps)),ch1,ch2,'UniformOutput',0);
        %         SL{trial} = cell(channels_len);
        cnt=0;
        
%         if idx_end-idx_start>=stp_range+w2+(embedding-1)*sampling_delay
        if data_len-idx_start>=trim_len
            channels=zeros(length(chans),trim_len);
            for i = 1:length(chans)
                channels(i,:)=rawdata_filtered{i}(idx_start:idx_start+trim_len-1);
            end
            %             clear('rawdata_filtered')
            if proc_avg==0
                SL_mexout=SL_mex(channels,w1,w2,embedding,sampling_delay,p_ref,stps,delays);
                SL_temp=cell(channels_len);
                SL_mat=reshape(SL_mexout,channels_len,channels_len,length(stps),length(delays));
                clear('SL_mexout');
                for m = 1:channels_len
                    for n = 1:channels_len
                        SL_temp{m,n}=squeeze(SL_mat(m,n,:,:))';
                    end
                end
                SL{trial}=SL_temp;
            else
                data_temp(:,:,trial)=channels;
            end
        end
        time_ind_back=trial;
%         if proc_avg==0
%             cwd=pwd;
%             cd('C:\')
%             save([fname '_temp'],'SL','time_ind_back');
%             system(['copy c:\' fname '_temp.mat "' cwd '"']);
%             system(['del c:\' fname '_temp.mat']);
%             cd(cwd);
%         end
    end
    multiWaitbar('Trial:',trial/trials,'color',[0.5,0.6,0.7]);
end

if proc_avg==1
    trial=1;
    channels=squeeze(mean(data_temp,3));
%     cnt=0;
    t0=toc;
    SL_mexout=SL_mex(channels,w1,w2,embedding,sampling_delay,p_ref,stps,delays);
    SL_temp=cell(channels_len);
    SL_mat=reshape(SL_mexout,channels_len,channels_len,length(stps),length(delays));
    clear('SL_mexout');
    for m = 1:channels_len
        for n = 1:channels_len
             SL_temp{m,n}=squeeze(SL_mat(m,n,:,:))';
        end
    end
    SL{trial}=SL_temp;
    t1=toc-t0;
    fprintf(2,[' time used ' num2str(t1) ' seconds \n']);
end
delete([fname '_temp.mat'])
SL_comp.data.SL=SL;
SL_comp.delay=(delays)*1000./system_fs+1;
SL_comp.startpoint=(stps)*1000./system_fs+1;
SL_comp.freq=[LP HS];
SL_comp.w1=w1;
SL_comp.w2=w2;
SL_comp.embedding=embedding;
SL_comp.sampling_delay=sampling_delay;
SL_comp.p_ref=p_ref;
SL_comp.fs=system_fs;

% k.delete;
end

function data_filtered=fir_filter(rawdata,LP,HS,fs)
if size(rawdata,2)==1
    rawdata=rawdata';
end
N=size(rawdata,2);
LP_d=floor(LP*N/fs)+1;
HP_d=round(HS*N/fs);
band_width=HP_d-LP_d;
LS=LP_d-0.1*band_width;
HS=HP_d+0.1*band_width;
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([LS LP_d HP_d HS]/(fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);

data_filtered=filter(Hd,rawdata);
end

function data_filtered=fft_filter(rawdata,LP,HS,fs)
if size(rawdata,2)==1
    rawdata=rawdata';
end
N=size(rawdata,2);
LP_d=floor(LP*N/fs)+1;
HS_d=round(HS*N/fs);
BP_d=zeros(size(rawdata));
BP_d(:,LP_d:HS_d)=1;
Y=fft(rawdata,N,2);  %得到原信号的频域表征
data_filtered=2*real(ifft(Y.*BP_d));
end
