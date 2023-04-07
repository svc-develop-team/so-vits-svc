#include "World/src/world/dio.h"
#include "World/src/world/stonemask.h"
#include "World/src/world/matlabfunctions.h"

class F0PreProcess
{
public:
	int fs;
	short hop;
	const int f0_bin = 256;
	const double f0_max = 1100.0;
	const double f0_min = 50.0;
	const double f0_mel_min = 1127.0 * log(1.0 + f0_min / 700.0);
	const double f0_mel_max = 1127.0 * log(1.0 + f0_max / 700.0);
	F0PreProcess(int sr = 16000, short h = 160) :fs(sr), hop(h) {}
	~F0PreProcess()
	{
		delete[] rf0;
		rf0 = nullptr;
	}
	void compute_f0(const double* audio, int64_t len);
	void InterPf0(int64_t len);
	long long* f0Log();
	int64_t getLen()const { return f0Len; }
private:
	double* rf0 = nullptr;
	int64_t f0Len = 0;
};

void F0PreProcess::compute_f0(const double* audio, int64_t len)
{
	DioOption Doption;
	InitializeDioOption(&Doption);
	Doption.f0_ceil = 800;
	Doption.frame_period = 1000.0 * hop / fs;
	f0Len = GetSamplesForDIO(fs, (int)len, Doption.frame_period);
	const auto tp = new double[f0Len];
	const auto tmpf0 = new double[f0Len];
	rf0 = new double[f0Len];
	Dio(audio, (int)len, fs, &Doption, tp, tmpf0);
	StoneMask(audio, (int)len, fs, tp, tmpf0, (int)f0Len, rf0);
	delete[] tmpf0;
	delete[] tp;
}

std::vector<double> arange(double start,double end,double step = 1.0,double div = 1.0)

void F0PreProcess::InterPf0(int64_t len)
{
	const auto xi = arange(0.0, (double)f0Len * (double)len, (double)f0Len, (double)len);
	const auto tmp = new double[xi.size() + 1];
	interp1(arange(0, (double)f0Len).data(), rf0, static_cast<int>(f0Len), xi.data(), (int)xi.size(), tmp);
	for (size_t i = 0; i < xi.size(); i++)
		if (isnan(tmp[i]))
			tmp[i] = 0.0;
	delete[] rf0;
    rf0 = nullptr;
	rf0 = tmp;
	f0Len = (int64_t)xi.size();
}

long long* F0PreProcess::f0Log()
{
	const auto tmp = new long long[f0Len];
	const auto f0_mel = new double[f0Len];
	for (long long i = 0; i < f0Len; i++)
	{
		f0_mel[i] = 1127 * log(1.0 + rf0[i] / 700.0);
		if (f0_mel[i] > 0.0)
			f0_mel[i] = (f0_mel[i] - f0_mel_min) * (f0_bin - 2.0) / (f0_mel_max - f0_mel_min) + 1.0;
		if (f0_mel[i] < 1.0)
			f0_mel[i] = 1;
		if (f0_mel[i] > f0_bin - 1)
			f0_mel[i] = f0_bin - 1;
		tmp[i] = (long long)round(f0_mel[i]);
	}
	delete[] f0_mel;
	delete[] rf0;
    rf0 = nullptr;
	return tmp;
}
