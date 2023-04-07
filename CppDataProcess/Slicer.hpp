#include <string>
#include <vector>
#include "Wav.hpp"

struct SliceResult
{
	std::vector<unsigned long long>	SliceOffset;
	std::vector<bool> SliceTag;
	cutResult(std::vector<unsigned long long>&& O, std::vector<bool>&& T) :SliceOffset(O), SliceTag(T) {}
};

double getAvg(const short* start, const short* end)
{
	const auto size = end - start + 1;
	auto avg = (double)(*start);
	for (auto i = 1; i < size; i++)
	{
		avg = avg + (abs((double)start[i]) - avg) / (double)(i + 1ull);
	}
	return avg;
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

