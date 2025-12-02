from mido import MidiFile
import mido
import time
import random
import copy
import numpy as np
from collections import defaultdict
mid = MidiFile('../PianoConcertoNo3Rachmaninoff.mid')

#print(f"音轨数: {len(mid.tracks)}, 每拍的 ticks 数: {mid.ticks_per_beat}")

#for i, track in enumerate(mid.tracks):
    #print(f"音轨 {i}: {track.name}")  # 打印音轨名称
    #for msg in track:
       # print(msg)  # 打印音轨中的每条消息

# 将 tick 转换为秒
#for msg in mid.play():
    #print(f"消息: {msg}, 在 {msg.time} ticks 后播放")
# 或者使用 tick2second 和 second2tick 函数进行转换

from mido import MidiFile, MidiTrack, Message, MetaMessage

# 1. 创建 MIDI 文件和音轨
#mid = MidiFile()
#track = MidiTrack()
#mid.tracks.append(track)

# 2. (可选)设置元数据，如速度、拍号、调号
#track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120))) # 设置速度为 120 BPM[citation:5]
#track.append(MetaMessage('time_signature', numerator=4, denominator=4)) # 设置拍号为 4/4 拍[citation:5]
#track.append(MetaMessage('key_signature', key='C')) # 设置调号为 C 大调[citation:5]

# 3. 设置乐器音色 (例如 program=0 通常代表大钢琴)
#track.append(Message('program_change', program=0, time=0))

# 4. 添加音符：按下中央 C，持续 1 拍 (假设 480 ticks/拍)
#track.append(Message('note_on', note=60, velocity=64, time=0))    # 立即按下
#track.append(Message('note_off', note=60, velocity=64, time=480)) # 在 480 ticks 后松开

# 5. 保存文件
mid.save('new_song.mid')

import random
import numpy as np
import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack


class RandomMusicGenerator:
    def __init__(self):
        self.segmentNum = 512
        self.segments = []
        self.timbreNum = 64
        self.timbre = [x for x in range(self.timbreNum)]
        self.segmentLength = 10  # 默认段落长度（小节数）
        self.minTrackNum = 1
        self.maxTrackNum = 16
        self.randCentre = 60  # 中央C (C4)
        self.randSigma = 10
        self.randOffset = 5
        self.noteNumMin = 15
        self.noteNumMax = 50
        self.defaultTrackNum = 3
        self.defaultTimbre = [0,10,24]
        self.timeResolution = 16  # 每拍的ticks数

        # 添加新成员变量
        self.l = 0.5  # 平滑度参数，控制与上一个音符的接近程度
        self.m = 0.5  # 中心倾向参数，控制向中心音符的接近程度

        # 节奏模式定义：名称, 节奏比例列表, 初始权重
        self.rhythmPatterns = [
            ("whole", [1.0], 10),  # 1个音 (100%)
            ("half_half", [0.5, 0.5], 10),  # 2个50%
            ("sixteenth_eighth", [0.25, 0.25, 0.5], 8),  # 前16后8
            ("eighth_sixteenth", [0.5, 0.25, 0.25], 8),  # 前8后16
            ("dotted1", [0.75, 0.25], 8),  # 附点1
            ("dotted2", [0.25, 0.75], 8),  # 附点2
            ("triplet", [1 / 3, 1 / 3, 1 / 3], 8),  # 三连音
            ("sixteenths", [0.25, 0.25, 0.25, 0.25], 10),  # 4个16分
            ("syncopation", [0.25, 0.5, 0.25], 8),  # 切分音
        ]

        # 钢琴键盘范围：A0到C8
        self.pianoMin = 21  # A0
        self.pianoMax = 108  # C8

        # 初始化段落的容器，但不在构造函数中生成音乐
        # self.segments 将在调用生成函数时填充

    def splitMidiAtMeasure(self, midi_file, split_measure):
        """
        将MIDI文件从小节之间切开

        参数:
        midi_file: MidiFile对象
        split_measure: 从第几个小节之后切开（从1开始计数）

        返回:
        两个MidiFile对象，分别是切开的前半部分和后半部分
        """
        # 创建两个新的MIDI文件
        first_half = MidiFile(ticks_per_beat=self.timeResolution)
        second_half = MidiFile(ticks_per_beat=self.timeResolution)

        # 计算切开位置的tick值
        ticks_per_measure = 4 * self.timeResolution  # 每小节的ticks数（4/4拍）
        split_tick = split_measure * ticks_per_measure

        # 遍历每个音轨
        for track_index, track in enumerate(midi_file.tracks):
            # 创建新的音轨
            first_track = MidiTrack()
            second_track = MidiTrack()

            # 复制元数据到两个新音轨
            for msg in track[:4]:  # 前4个消息通常是元数据
                first_track.append(copy.deepcopy(msg))
                second_track.append(copy.deepcopy(msg))

            # 遍历音符事件
            current_tick = 0
            active_notes = {}  # 记录激活的音符

            for msg in track[4:]:  # 从第5个消息开始通常是音符事件
                if msg.type == 'note_on' or msg.type == 'note_off':
                    current_tick += msg.time

                    if current_tick < split_tick:
                        # 添加到前半部分
                        first_track.append(copy.deepcopy(msg))

                        # 记录激活的音符
                        if msg.type == 'note_on' and msg.velocity > 0:
                            active_notes[msg.note] = current_tick
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            if msg.note in active_notes:
                                del active_notes[msg.note]
                    else:
                        # 添加到后半部分，调整时间
                        adjusted_tick = current_tick - split_tick

                        # 创建调整后的消息
                        adjusted_msg = copy.deepcopy(msg)
                        adjusted_msg.time = adjusted_tick

                        second_track.append(adjusted_msg)

                        # 记录激活的音符
                        if msg.type == 'note_on' and msg.velocity > 0:
                            active_notes[msg.note] = current_tick
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            if msg.note in active_notes:
                                del active_notes[msg.note]
                else:
                    # 处理其他类型的消息（如控制变化、音色变化等）
                    if current_tick < split_tick:
                        first_track.append(copy.deepcopy(msg))
                    else:
                        adjusted_msg = copy.deepcopy(msg)
                        adjusted_msg.time = current_tick - split_tick
                        second_track.append(adjusted_msg)

            # 确保前半部分结束所有激活的音符
            for note in list(active_notes.keys()):
                if active_notes[note] < split_tick:
                    # 添加note_off事件
                    note_off = Message('note_off', note=note, velocity=64, time=0)
                    first_track.append(note_off)

            # 添加音轨
            first_half.tracks.append(first_track)
            second_half.tracks.append(second_track)

        return first_half, second_half

    def concatenateMidiFiles(self, midi_files):
        """
        连接多个MIDI文件

        参数:
        midi_files: MidiFile对象的列表

        返回:
        一个连接后的MidiFile对象

        要求:
        所有MIDI文件必须有相同的音轨数，且每个音轨有相同的元数据
        """
        if not midi_files:
            return None

        # 检查音轨数是否一致
        track_count = len(midi_files[0].tracks)
        for i, midi_file in enumerate(midi_files[1:], 1):
            if len(midi_file.tracks) != track_count:
                raise ValueError(f"MIDI文件 {i} 的音轨数不一致: 期望 {track_count}, 实际 {len(midi_file.tracks)}")

        # 检查每个音轨的元数据是否一致
        for track_index in range(track_count):
            reference_meta = []
            for msg in midi_files[0].tracks[track_index]:
                if msg.is_meta:
                    reference_meta.append((msg.type, getattr(msg, 'numerator', None),
                                           getattr(msg, 'denominator', None), getattr(msg, 'key', None)))

            for i, midi_file in enumerate(midi_files[1:], 1):
                current_meta = []
                for msg in midi_file.tracks[track_index]:
                    if msg.is_meta:
                        current_meta.append((msg.type, getattr(msg, 'numerator', None),
                                             getattr(msg, 'denominator', None), getattr(msg, 'key', None)))

                if reference_meta != current_meta:
                    raise ValueError(f"MIDI文件 {i} 的音轨 {track_index} 的元数据不一致")

        # 创建新的MIDI文件
        concatenated = MidiFile(ticks_per_beat=self.timeResolution)

        # 对每个音轨进行连接
        for track_index in range(track_count):
            new_track = MidiTrack()

            # 复制第一个文件的元数据
            for msg in midi_files[0].tracks[track_index]:
                if msg.is_meta:
                    new_track.append(copy.deepcopy(msg))

            # 连接所有文件的音符事件
            last_note_off_time = {}  # 记录每个音符的最后关闭时间
            current_absolute_time = 0

            for file_index, midi_file in enumerate(midi_files):
                # 计算文件的绝对时间偏移
                if file_index > 0:
                    # 在前一个文件结束后添加一个小节的分隔
                    separator_time = 4 * self.timeResolution  # 一个小节的ticks
                    separator_msg = Message('note_off', note=0, velocity=0, time=separator_time)
                    new_track.append(separator_msg)
                    current_absolute_time += separator_time

                # 处理当前文件的音轨
                track = midi_file.tracks[track_index]
                track_absolute_time = 0

                for msg in track:
                    if not msg.is_meta:  # 跳过元数据，只处理音符事件
                        track_absolute_time += msg.time

                        if msg.type == 'note_on' or msg.type == 'note_off':
                            # 创建新的消息，调整时间
                            new_msg = copy.deepcopy(msg)
                            new_msg.time = track_absolute_time
                            new_track.append(new_msg)

                            # 更新音符状态
                            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                                last_note_off_time[msg.note] = current_absolute_time + track_absolute_time
                            elif msg.type == 'note_on' and msg.velocity > 0:
                                if msg.note in last_note_off_time:
                                    # 确保音符不会重叠
                                    if current_absolute_time + track_absolute_time < last_note_off_time[msg.note]:
                                        # 需要插入一个note_off事件
                                        note_off = Message('note_off', note=msg.note, velocity=64,
                                                           time=last_note_off_time[msg.note] - (
                                                                       current_absolute_time + track_absolute_time))
                                        new_track.append(note_off)

                current_absolute_time += track_absolute_time

            # 添加音轨到连接后的文件
            concatenated.tracks.append(new_track)

        return concatenated

    def generateSegments(self, numSegments=None, segmentLength=None):
        """
        生成多个段落（小节）

        参数:
        numSegments: 要生成的段落数量，如果为None则使用self.segmentNum
        segmentLength: 每个段落的长度（小节数），如果为None则使用self.segmentLength
        """
        if numSegments is None:
            numSegments = self.segmentNum

        if segmentLength is None:
            segmentLength = self.segmentLength

        self.segments = []  # 清空现有的段落

        for i in range(numSegments):
            # 生成指定长度的段落
            segment = self.generateSegment(segmentLength)
            self.segments.append(segment)

        print(f"Generated {numSegments} segments, each with {segmentLength} measures")
        return self.segments

    def generateSegment(self, lengthInMeasures):
        """
        生成单个指定长度的段落

        参数:
        lengthInMeasures: 段落的长度（小节数）

        返回:
        一个MidiFile对象，包含指定小节数的音乐
        """
        # 创建MIDI文件
        segment = MidiFile(ticks_per_beat=self.timeResolution)

        # 随机确定音轨数量
        trackNum = self.defaultTrackNum if self.defaultTrackNum != 0 else random.randint(self.minTrackNum, self.maxTrackNum)
        # 创建音轨
        for j in range(trackNum):
            track = MidiTrack()
            segment.tracks.append(track)

            # 添加音轨元信息
            track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
            track.append(MetaMessage('time_signature', numerator=4, denominator=4))
            track.append(MetaMessage('key_signature', key='C'))
            track.append(Message('program_change',
                                 program=self.defaultTimbre[j] if self.defaultTimbre != [] else self.timbre[random.randint(0, self.timbreNum - 1)],
                                 time=0))

            # 为该音轨生成音符
            self.generateNotesForTrack(track, lengthInMeasures)

        return segment

    def generateNotesForTrack(self, track, numMeasures):
        """为单个音轨生成指定小节数的音符"""
        # 设置初始音高
        prevPitch = self.randCentre

        # 一个小节的总ticks数 (4拍 * 每拍的ticks数)
        measureTicks = 4 * self.timeResolution

        # 当前时间（从0开始）
        curTime = 0

        # 对于每个小节
        for measureIndex in range(numMeasures):
            # 确定这个小节要生成的音符数量
            notesInMeasure = random.randint(
                max(1, self.noteNumMin // 4),  # 至少1个音符
                max(2, self.noteNumMax // 4)  # 至少2个音符
            )

            # 生成这个小节的音符
            for i in range(notesInMeasure):
                # 选择节奏模式
                rhythmPattern = self.selectRhythmPattern(measureTicks)
                patternName, ratios, _ = rhythmPattern

                # 将节奏比例转换为ticks
                durations = [int(ratio * measureTicks) for ratio in ratios]

                # 对每个音符段生成音符
                for duration in durations:
                    if duration <= 0:
                        continue

                    # 检查是否应该生成休止符（有一定概率）
                    if random.random() < 0.1:  # 10%的概率休止
                        curTime += duration
                        continue

                    # 生成音高
                    pitch = self.generatePitchWithWeight(prevPitch)
                    prevPitch = pitch

                    # 确保音高在钢琴范围内
                    pitch = max(self.pianoMin, min(self.pianoMax, pitch))

                    # 生成随机力度
                    velocity = random.randint(60, 100)

                    # 添加音符开启事件
                    track.append(Message('note_on', note=pitch,
                                         velocity=velocity, time=0))

                    # 添加音符关闭事件（音符持续时间为节奏段的duration）
                    track.append(Message('note_off', note=pitch,
                                         velocity=velocity, time=duration))

                    # 更新当前时间
                    curTime += duration

            # 小节结束，重置curTime为0（MIDI时间通常是相对的）
            # 注意：在MIDI中，每个事件的time是相对于上一个事件的时间
            # 所以不需要手动管理curTime，除非我们需要特定的时间间隔
            # 这里我们保持原逻辑，实际上每个音符的time已经正确设置了

    def generateSpecificSegment(self, segmentId, lengthInMeasures):
        """
        生成特定ID的段落，或替换现有段落

        参数:
        segmentId: 段落的ID（索引）
        lengthInMeasures: 段落的长度（小节数）

        返回:
        生成的MidiFile对象
        """
        # 如果segments列表不够长，扩展它
        while len(self.segments) <= segmentId:
            self.segments.append(None)

        # 生成新的段落
        segment = self.generateSegment(lengthInMeasures)

        # 替换或添加到segments列表
        self.segments[segmentId] = segment

        return segment

    def generateVariableLengthSegments(self, lengths):
        """
        生成多个不同长度的段落

        参数:
        lengths: 长度列表，每个元素是一个整数，表示对应段落的小节数

        返回:
        生成的MidiFile对象列表
        """
        self.segments = []

        for i, length in enumerate(lengths):
            print(f"Generating segment {i} with {length} measures...")
            segment = self.generateSegment(length)
            self.segments.append(segment)

        print(f"Generated {len(lengths)} segments with variable lengths")
        return self.segments

    def selectRhythmPattern(self, measureTicks):
        """根据权重选择合适的节奏模式"""
        weights = []
        validPatterns = []

        # 计算每个模式的权重，排除会生成过短音符的模式
        for patternName, ratios, baseWeight in self.rhythmPatterns:
            minDuration = min(ratios) * measureTicks

            # 检查最小持续时间是否大于最小分辨率
            if minDuration >= 1:  # 至少1个tick
                # 根据音符数量调整权重：音符越少，权重越高
                noteCountWeight = 1.0 / len(ratios)
                adjustedWeight = baseWeight * noteCountWeight

                weights.append(adjustedWeight)
                validPatterns.append((patternName, ratios, baseWeight))
            else:
                # 跳过会生成过短音符的模式
                continue

        # 如果没有有效的模式，使用默认的whole模式
        if not validPatterns:
            return ("whole", [1.0], 10)

        # 根据权重随机选择
        pattern = random.choices(validPatterns, weights=weights, k=1)[0]
        return pattern

    def generatePitchWithWeight(self, prevPitch):
        """根据权重生成音高"""
        # 候选音高范围（上一个音符周围±12个半音）
        minPitch = max(self.pianoMin, prevPitch - 12)
        maxPitch = min(self.pianoMax, prevPitch + 12)

        # 计算每个候选音高的权重
        candidates = list(range(minPitch, maxPitch + 1))
        weights = []

        for pitch in candidates:
            # 计算两个高斯项的权重
            prevTerm = ((pitch - prevPitch) ** 2) / (self.randOffset ** 2)
            centerTerm = ((pitch - self.randCentre) ** 2) / (self.randSigma ** 2)

            # 计算总权重
            weight = np.exp(-self.l * prevTerm - self.m * centerTerm)
            weights.append(weight)

        # 归一化权重
        totalWeight = sum(weights)
        if totalWeight > 0:
            normalizedWeights = [w / totalWeight for w in weights]
        else:
            normalizedWeights = [1.0 / len(weights)] * len(weights)

        # 根据权重随机选择音高
        pitch = random.choices(candidates, weights=normalizedWeights, k=1)[0]
        return pitch

    def saveSegments(self, outputDir="output"):
        """保存所有段落为MIDI文件"""
        import os
        os.makedirs(outputDir, exist_ok=True)

        for i, segment in enumerate(self.segments):
            if segment is not None:
                filename = f"{outputDir}/segment_{i:03d}.mid"
                segment.save(filename)
                print(f"Saved {filename}")


def linear(i, total):
    return i / total


def firstHalf(i, total):
    return 1 if i < total // 2 else 0


def rand(segment):
    return random.randint(0, 10)


def count_notes_adaptation(segment):
    """计算音符数量作为适应度"""
    note_count = 0
    for track in segment.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count += 1
    return note_count


def pitch_variance_adaptation(segment):
    """计算音高方差作为适应度（方差越小越好）"""
    pitches = []
    for track in segment.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)

    if len(pitches) > 1:
        variance = np.var(pitches)
        return 1.0 / (1.0 + variance)  # 方差越小，适应度越高
    return 0.0

def shift_up_or_down_prob_0_01(segment):
    shift = random.randint(-1,2)
    curnote = segment.tracks[0].note
    modified = 0
    for track in segment.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and random.randint(0,100) <= 1:
                curnote = msg.note
                modified = 1
                msg.note += shift
            if msg.type == 'note_off' and msg.velocity > 0 and msg.note == curnote and modified:
                modified = 0
                msg.note += shift


def note_number_to_name(note_number):
    """将MIDI音符编号转换为音高名称"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_number // 12 - 1
    note_name = notes[note_number % 12]
    return f"{note_name}{octave}"

def rhythmic_complexity_adaptation(segment, time_resolution=16):
    """计算节奏复杂度作为适应度"""
    # 统计不同节奏类型的分布
    rhythm_types = defaultdict(int)
    total_notes = 0

    for track in segment.tracks:
        current_time = 0
        note_on_times = {}

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                # 记录音符开始时间
                note_on_times[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_on_times:
                    # 计算音符持续时间
                    duration = current_time - note_on_times[msg.note]
                    # 将持续时间分类为节奏类型
                    if duration <= time_resolution // 4:  # 十六分音符
                        rhythm_types['sixteenth'] += 1
                    elif duration <= time_resolution // 2:  # 八分音符
                        rhythm_types['eighth'] += 1
                    elif duration <= time_resolution:  # 四分音符
                        rhythm_types['quarter'] += 1
                    elif duration <= time_resolution * 2:  # 二分音符
                        rhythm_types['half'] += 1
                    else:  # 全音符或更长
                        rhythm_types['whole'] += 1

                    total_notes += 1
                    del note_on_times[msg.note]

    if total_notes == 0:
        return 0.0

    # 计算节奏类型分布的熵（多样性越高，熵越大）
    entropy = 0.0
    for count in rhythm_types.values():
        probability = count / total_notes
        entropy -= probability * np.log2(probability) if probability > 0 else 0

    # 归一化熵值（最大熵是log2(5)≈2.32，因为我们有5种节奏类型）
    max_entropy = np.log2(5)
    return entropy / max_entropy


class Iterator:
    def __init__(self):
        self.rmg = RandomMusicGenerator()
        self.segmentPoolSize = 1024
        self.segmentLength = 64
        self.segmentPool = self.rmg.generateSegments(self.segmentPoolSize, self.segmentLength)
        self.keepFunction = firstHalf
        self.adaptationFunction = pitch_variance_adaptation  # 使用音高方差作为适应度
        self.mutationPool = [shift_up_or_down_prob_0_01]
        self.mutationIntensity = self.segmentPoolSize // 8
        self.crossIntensity = self.segmentPoolSize // 4

    def analyze_segment(self, segment, segment_id):
        """详细分析一个MIDI段落"""
        print(f"\n{'=' * 60}")
        print(f"段落下标: {segment_id}")
        print(f"音轨数: {len(segment.tracks)}")

        # 计算总音符数
        total_notes = 0
        total_duration = 0

        # 分析每个音轨
        for track_idx, track in enumerate(segment.tracks):
            print(f"\n音轨 {track_idx}:")

            # 获取音轨元数据
            tempo = None
            time_signature = None
            key_signature = None
            program = None

            for msg in track:
                if msg.is_meta:
                    if msg.type == 'set_tempo':
                        tempo = mido.tempo2bpm(msg.tempo)
                    elif msg.type == 'time_signature':
                        time_signature = f"{msg.numerator}/{msg.denominator}"
                    elif msg.type == 'key_signature':
                        key_signature = msg.key
                    elif msg.type == 'program_change':
                        program = msg.program

            print(f"  速度: {tempo} BPM" if tempo else "  速度: 未知")
            print(f"  拍号: {time_signature}" if time_signature else "  拍号: 未知")
            print(f"  调号: {key_signature}" if key_signature else "  调号: 未知")
            print(f"  乐器: {program}" if program is not None else "  乐器: 未知")

        # 按小节分析音符
        print("\n按小节分析:")
        ticks_per_measure = 4 * self.rmg.timeResolution
        measures = []

        for measure_idx in range(min(self.segmentLength, 16)):  # 只分析前16个小节，避免输出太长
            measure_start = measure_idx * ticks_per_measure
            measure_end = (measure_idx + 1) * ticks_per_measure
            measure_notes = []

            # 分析每个音轨在当前小节的音符
            for track_idx, track in enumerate(segment.tracks):
                current_time = 0

                # 重新遍历音轨，计算绝对时间
                abs_time = 0
                active_notes = {}  # 音符号 -> 开始时间

                for msg in track:
                    abs_time += msg.time

                    # 如果是音符开始事件
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if measure_start <= abs_time < measure_end:
                            # 音符在小节内开始
                            note_info = {
                                'note': msg.note,
                                'velocity': msg.velocity,
                                'track': track_idx,
                                'start_time': abs_time - measure_start,
                                'end_time': None
                            }
                            measure_notes.append(note_info)
                            active_notes[msg.note] = len(measure_notes) - 1

                    # 如果是音符结束事件
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in active_notes:
                            note_idx = active_notes[msg.note]
                            if measure_start <= abs_time < measure_end:
                                # 音符在同一小节内结束
                                measure_notes[note_idx]['end_time'] = abs_time - measure_start
                                measure_notes[note_idx]['duration'] = abs_time - measure_notes[note_idx]['start_time']
                            else:
                                # 音符延续到下一个小节
                                measure_notes[note_idx]['end_time'] = ticks_per_measure
                                measure_notes[note_idx]['duration'] = ticks_per_measure - measure_notes[note_idx][
                                    'start_time']
                            del active_notes[msg.note]

            # 统计小节的音符信息
            if measure_notes:
                notes = [n['note'] for n in measure_notes]
                pitches = [note_number_to_name(n) for n in notes]
                avg_pitch = np.mean(notes) if notes else 0

                print(f"  小节 {measure_idx + 1}:")
                print(f"    音符数: {len(measure_notes)}")
                print(f"    平均音高: {avg_pitch:.1f} ({note_number_to_name(int(avg_pitch)) if notes else 'N/A'})")
                print(f"    音高范围: {min(notes) if notes else 'N/A'} - {max(notes) if notes else 'N/A'}")

                # 只显示前8个音高，避免输出太长
                if pitches:
                    print(f"    音高列表: {', '.join(pitches[:8])}{'...' if len(pitches) > 8 else ''}")

                # 统计音高分布
                pitch_counts = defaultdict(int)
                for pitch in pitches:
                    pitch_counts[pitch] += 1

                # 打印最常见的音高
                if pitch_counts:
                    most_common = sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    常见音高: {', '.join([f'{pitch}({count})' for pitch, count in most_common])}")

                # 显示节奏信息
                if measure_notes:
                    durations = [n['duration'] / (ticks_per_measure / 4) for n in measure_notes if 'duration' in n]
                    if durations:
                        print(f"    平均音符时长: {np.mean(durations):.2f} 拍")

            measures.append(measure_notes)
            total_notes += len(measure_notes)

        print(f"\n总结:")
        print(f"  总音符数: {total_notes}")
        print(f"  段落长度: {self.segmentLength} 小节")

        # 计算并显示多种适应度
        fitness_pitch = pitch_variance_adaptation(segment)
        fitness_rhythm = rhythmic_complexity_adaptation(segment, self.rmg.timeResolution)
        fitness_notes = count_notes_adaptation(segment) / 1000  # 归一化

        print(f"  音高方差适应度: {fitness_pitch:.4f}")
        print(f"  节奏复杂度适应度: {fitness_rhythm:.4f}")
        print(f"  音符数量适应度: {fitness_notes:.4f}")
        print(f"  综合适应度: {self.adaptationFunction(segment):.4f}")

        return measures, total_notes

    def output_top_segments(self, top_n=10):
        """输出适应度最高的n个段落的详细信息"""
        print("\n" + "=" * 80)
        print(f"输出适应度最高的 {top_n} 个段落")
        print("=" * 80)

        # 计算所有段落的适应度
        segments_with_fitness = []
        for idx, segment in enumerate(self.segmentPool):
            fitness = self.adaptationFunction(segment)
            segments_with_fitness.append((fitness, idx, segment))

        # 按适应度排序（从高到低）
        segments_with_fitness.sort(reverse=True, key=lambda x: x[0])

        # 输出前n个
        for rank, (fitness, idx, segment) in enumerate(segments_with_fitness[:top_n], 1):
            print(f"\n{'=' * 60}")
            print(f"排名 #{rank}")
            print(f"适应度: {fitness:.4f}")
            print(f"段落下标: {idx}")

            # 保存为文件
            output_filename = f"top_segment_{rank:03d}.mid"
            segment.save(output_filename)
            print(f"已保存为: {output_filename}")

            # 详细分析
            self.analyze_segment(segment, idx)

        print("\n" + "=" * 80)
        print("输出完成！")

        # 返回排名信息
        return [(rank, fitness, idx) for rank, (fitness, idx, _) in enumerate(segments_with_fitness[:top_n], 1)]

    def output_statistics(self):
        """输出统计信息"""
        print("\n" + "=" * 80)
        print("迭代统计信息")
        print("=" * 80)

        # 计算适应度统计
        fitness_values = [self.adaptationFunction(seg) for seg in self.segmentPool]

        if fitness_values:
            print(f"段落总数: {len(self.segmentPool)}")
            print(f"最高适应度: {max(fitness_values):.4f}")
            print(f"最低适应度: {min(fitness_values):.4f}")
            print(f"平均适应度: {np.mean(fitness_values):.4f}")
            print(f"适应度标准差: {np.std(fitness_values):.4f}")

            # 适应度分布
            hist, bins = np.histogram(fitness_values, bins=10)
            print("\n适应度分布:")
            for i in range(len(hist)):
                print(f"  {bins[i]:.2f}-{bins[i + 1]:.2f}: {hist[i]} 段落")

        # 计算音符统计
        total_notes = []
        for segment in self.segmentPool:
            notes = 0
            for track in segment.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        notes += 1
            total_notes.append(notes)

        if total_notes:
            print(f"\n音符统计:")
            print(f"  总音符数范围: {min(total_notes)} - {max(total_notes)}")
            print(f"  平均音符数: {np.mean(total_notes):.1f}")

            # 显示顶级段落的音符数
            if len(total_notes) >= 10:
                sorted_indices = np.argsort(total_notes)[::-1]  # 从大到小排序
                print(f"\n音符最多的5个段落:")
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    print(f"  第{i + 1}名: 段落{idx} - {total_notes[idx]} 个音符")

    def mutate(self):
        l1 = len(self.segmentPool)
        l2 = len(self.mutationPool)
        for i in range(self.mutationIntensity):
            n = random.randint(0, l1 - 1)
            m = random.randint(0, l2 - 1)
            # 这里需要修改，因为mutationPool目前是空的
            # self.segmentPool[n] = self.mutationPool[m](self.segmentPool[n])
            pass

    def cross(self):
        l = len(self.segmentPool)
        for i in range(self.crossIntensity):
            x = random.randint(0, l - 1)
            y = random.randint(0, l - 1)
            n = random.randint(1, self.segmentLength - 1)  # 避免在边界切开
            try:
                lx, rx = self.rmg.splitMidiAtMeasure(self.segmentPool[x], n)
                ly, ry = self.rmg.splitMidiAtMeasure(self.segmentPool[y], n)
                self.segmentPool[x] = self.rmg.concatenateMidiFiles([lx, ry])
                self.segmentPool[y] = self.rmg.concatenateMidiFiles([ly, rx])
            except Exception as e:
                print(f"交叉操作出错: {e}")
                # 如果出错，跳过这次交叉
                continue

    def iteratesingle(self):
        l1 = len(self.segmentPool)
        temp = self.segmentPool.copy()

        # 执行变异和交叉
        self.mutate()
        self.cross()

        # 合并原始和变异后的段落
        temp += self.segmentPool

        # 计算适应度并排序
        temp_with_fitness = []
        for seg in temp:
            try:
                fitness = self.adaptationFunction(seg)
                temp_with_fitness.append((fitness, seg))
            except Exception as e:
                print(f"计算适应度出错: {e}")
                continue

        temp_with_fitness.sort(reverse=True, key=lambda x: x[0])

        # 选择保留的段落
        l = len(temp_with_fitness)
        res = []
        for i in range(l):
            if self.keepFunction(i, l) == 1:
                res.append(temp_with_fitness[i][1])

        # 确保不超过池大小
        self.segmentPool = res[:self.segmentPoolSize]

        # 如果不足，用随机段落填充
        while len(self.segmentPool) < self.segmentPoolSize:
            self.segmentPool.append(self.rmg.generateSegment(self.segmentLength))

    def iteration(self, n):
        for i in range(n):
            print(f"\n迭代 #{i + 1}/{n}")
            self.iteratesingle()
            # 每5次迭代输出一次统计
            if (i + 1) % 5 == 0:
                self.output_statistics()

    def output(self):
        """输出所有信息"""
        print("\n" + "=" * 80)
        print("最终输出")
        print("=" * 80)

        # 输出统计信息
        self.output_statistics()

        # 输出顶级段落
        top_segments = self.output_top_segments(top_n=10)

        # 保存所有段落
        import os
        output_dir = "final_output"
        os.makedirs(output_dir, exist_ok=True)

        for idx, segment in enumerate(self.segmentPool):
            filename = f"{output_dir}/segment_{idx:04d}.mid"
            segment.save(filename)

        print(f"\n所有段落已保存到 {output_dir} 目录")

        return top_segments


# 使用示例
if __name__ == "__main__":
    print("初始化迭代器...")
    iterator = Iterator()

    # 运行迭代
    print("开始迭代...")
    iterator.iteration(10)  # 运行10次迭代，避免太长

    # 输出最终结果
    print("\n生成最终输出...")
    top_segments = iterator.output()

    print(f"\n迭代完成！适应度最高的段落已保存。")

    # 显示顶级段落的总结
    print("\n顶级段落总结:")
    print("排名 | 适应度 | 段落下标 | 文件名")
    print("-" * 50)
    for rank, fitness, idx in top_segments[:5]:  # 只显示前5个
        print(f"#{rank:2d}  | {fitness:.4f} | {idx:8d} | top_segment_{rank:03d}.mid")

    print("\n程序完成！")