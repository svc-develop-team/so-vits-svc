#include "World/src/world/dio.h"
#include "World/src/world/stonemask.h"
#include "World/src/world/matlabfunctions.h"

struct SliceResult
{
	std::vector<unsigned long long>	SliceOffset;
	std::vector<bool> SliceTag;
	cutResult(std::vector<unsigned long long>&& O, std::vector<bool>&& T) :SliceOffset(O), SliceTag(T) {}
};

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
	std::vector<long long> GetF0AndOtherInput(const double* audio, int64_t audioLen, int64_t hubLen, int64_t tran);
	
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
{
	std::vector<double> output;
	while(start<end)
	{
		output.push_back(start / div);
		start += step;
	}
	return output;
}

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

inline SliceResult SliceWav(Wav& input, double threshold, unsigned long minLen, unsigned short frame_len, unsigned short frame_shift)
{
	const auto header = input.getHeader();
	if (header.Subchunk2Size < minLen * header.bytesPerSec)
		return { {0,header.Subchunk2Size},{true} };
	auto ptr = input.getData();
	std::vector<unsigned long long> output;
	std::vector<bool> tag;
	auto n = (header.Subchunk2Size / frame_shift) - 2 * (frame_len / frame_shift);
	unsigned long nn = 0;
	bool cutTag = true;
	output.emplace_back(0);
	while (n--)
	{
		//if (nn > minLen * header.bytesPerSec)
		if (cutTag)
		{
			const auto vol = abs(getAvg((short*)ptr, (short*)ptr + frame_len));
			if (vol < threshold)
			{
				cutTag = false;
				if (nn > minLen * header.bytesPerSec)
				{
					nn = 0;
					output.emplace_back((ptr - input.getData()) + (frame_len / 2));
				}
			}
			else
			{
				cutTag = true;
			}
		}
		else
		{
			const auto vol = abs(getAvg((short*)ptr, (short*)ptr + frame_len));
			if (vol < threshold)
			{
				cutTag = false;
			}
			else
			{
				cutTag = true;
				if (nn > minLen * header.bytesPerSec)
				{
					nn = 0;
					output.emplace_back((ptr - input.getData()) + (frame_len / 2));
				}
			}
		}
		nn += frame_shift;
		ptr += frame_shift;
	}
	output.push_back(header.Subchunk2Size);
	for (size_t i = 1; i < output.size(); i++)
	{
		tag.push_back(abs(getAvg((short*)(input.getData() + output[i - 1]), (short*)(input.getData() + output[i]))) > threshold);
	}
	return { std::move(output),std::move(tag) };
}

