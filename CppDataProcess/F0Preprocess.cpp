#include "F0Preprocess.hpp"


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

std::vector<long long> F0PreProcess::GetF0AndOtherInput(const double* audio, int64_t audioLen, int64_t hubLen, int64_t tran)
{
	compute_f0(audio, audioLen);
	for (int64_t i = 0; i < f0Len; ++i)
	{
		rf0[i] = rf0[i] * pow(2.0, static_cast<double>(tran) / 12.0);
		if (rf0[i] < 0.001)
			rf0[i] = NAN;
	}
	InterPf0(hubLen);
	const auto O0f = f0Log();
	std::vector<long long> Of0(O0f, O0f + f0Len);
    delete[] O0f;
	return Of0;
}

std::vector<long long> getAligments(size_t specLen, size_t hubertLen)
{
	std::vector<long long> mel2ph(specLen + 1, 0);

	size_t startFrame = 0;
	const double ph_durs = static_cast<double>(specLen) / static_cast<double>(hubertLen);
	for (size_t iph = 0; iph < hubertLen; ++iph)
	{
		const auto endFrame = static_cast<size_t>(round(static_cast<double>(iph) * ph_durs + ph_durs));
		for (auto j = startFrame; j < endFrame + 1; ++j)
			mel2ph[j] = static_cast<long long>(iph) + 1;
		startFrame = endFrame + 1;
	}

	return mel2ph;
}

std::vector<float> F0PreProcess::GetF0AndOtherInputF0(const double* audio, int64_t audioLen, int64_t tran)
{
	compute_f0(audio, audioLen);
	for (int64_t i = 0; i < f0Len; ++i)
	{
		rf0[i] = log2(rf0[i] * pow(2.0, static_cast<double>(tran) / 12.0));
		if (rf0[i] < 0.001)
			rf0[i] = NAN;
	}
	const int64_t specLen = audioLen / hop;
	InterPf0(specLen);

    std::vector<float> Of0(specLen, 0.0);

    double last_value = 0.0;
    for (int64_t i = 0; i < specLen; ++i)
    {
        if (rf0[i] <= 0.0)
        {
            int64_t j = i + 1;
            for (; j < specLen; ++j)
            {
                if (rf0[j] > 0.0)
                    break;
            }
            if (j < specLen - 1)
            {
                if (last_value > 0.0)
                {
                    const auto step = (rf0[j] - rf0[i - 1]) / double(j - i);
                    for (int64_t k = i; k < j; ++k)
                        Of0[k] = float(rf0[i - 1] + step * double(k - i + 1));
                }
                else
                    for (int64_t k = i; k < j; ++k)
                        Of0[k] = float(rf0[j]);
                i = j;
            }
            else
            {
                for (int64_t k = i; k < specLen; ++k)
                    Of0[k] = float(last_value);
                i = specLen;
            }
        }
        else
        {
            Of0[i] = float(rf0[i - 1]);
            last_value = rf0[i];
        }
    }
    delete[] rf0;
    rf0 = nullptr;
	return Of0;
}
