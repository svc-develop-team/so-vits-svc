#include "world/dio.h"
#include "world/stonemask.h"
#include "world/matlabfunctions.h"
#include <string>
#include <vector>

//Cpp F0 Preprocess

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
	std::vector<float> GetF0AndOtherInputF0(const double* audio, int64_t audioLen, int64_t tran);
private:
	double* rf0 = nullptr;
	int64_t f0Len = 0;
};

std::vector<long long> getAligments(size_t specLen, size_t hubertLen);
