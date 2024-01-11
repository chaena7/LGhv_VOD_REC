from django.shortcuts import render, get_object_or_404,redirect
from rest_framework.views import APIView
from .serializers import *
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer, TokenRefreshSerializer
from rest_framework import status
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse

from .models import Userinfo, VOD_INFO, VODSumut, VODLog, CONlog
from django.db.models import Q, F, Sum, Count, Subquery
from django.db.models.functions import Cast
from django.db.models import FloatField
import jwt

from django.contrib.auth import authenticate, get_user_model
from base.settings import SECRET_KEY
from rest_framework_simplejwt.tokens import RefreshToken
# from django.contrib.auth.hashers import check_password
import socket

from rest_framework.decorators import api_view
from itertools import chain, groupby
from operator import itemgetter
import random
from .forms import LoginForm
import re

#form 사용법
# @api_view(['POST', 'GET'])
# def test(request):
#     if request.method == 'POST':
#         form = LoginForm(request.POST)
#         if form.is_valid():
#             user = Userinfo()
#             user.subsr = form.cleaned_data['subsr']
#             user.is_active = 1
#             user.save()
#             return redirect('admin/')
#     else:
#         form = LoginForm()
#         context = {'form': form}
#         return render(request, 'test.html', context)
    # else:
        # return Response({"message":"method err"}, status= status.HTTP_400_BAD_REQUEST)

#login 성공시 
@api_view(('GET',))
def login_suc(request):
    if request.method == 'GET':
        access = request.COOKIES['access']
        payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
        pk = payload.get('user_id')
        user = get_object_or_404(Userinfo, pk = pk)
        serializer = UserSerializer(instance = user)
        user_id = serializer.data.get('subsr', None)
        return Response({"user":user_id}, status= status.HTTP_200_OK)


#login 상태 확인
@api_view(('GET',))
def status_check(request):
    # if request.method == 'POST':
        # return redirect('login')
    if request.method == 'GET':
        try:
            access = request.COOKIES['access']
        except KeyError:
             return Response({'error': 'KeyError'}, status = status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            try:
                payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
                pk = payload.get('user_id')
                user = get_object_or_404(Userinfo, pk = pk)
                serializer = UserSerializer(instance = user)
                user_id = serializer.data.get('subsr', None)
                if user_id != None:
                    return Response({'msg':'login suc'}, status = status.HTTP_200_OK)
                else:
                    return Response({"error": 'login first'}, status = status.HTTP_400_BAD_REQUEST)
            except jwt.InvalidSignatureError:
                return Response({'error':'invalid signature'}, status = status.HTTP_400_BAD_REQUEST)
            except(jwt.exceptions.ExpiredSignatureError):
            #token 만료 시 갱신
                print(1)
                data = {'refresh': request.COOKIES.get('refresh', None)}
                serializer = TokenObtainPairSerializer(data = data)
                if serializer.is_valid(raise_exception=True):
                    access = serializer.data.get('access', None)
                    refresh = serializer.data.get('refresh', None)
                    payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
                    pk = payload.get('userinfo_subsr')
                    user = get_object_or_404(Userinfo, pk = pk)
                    serializer = UserSerializer(instance = user)
                    res = Response(serializer.data, status=status.HTTP_200_OK)
                    res.set_cookie('access', access)
                    res.set_cookie('refresh', refresh)
                    return res
                return Response({'error':'invalid token'}, status = status.HTTP_400_BAD_REQUEST)
            # except(jwt.exceptions.InvalidTokenError):
            #     #사용 불가 토큰인 경우
            #     return redirect(request.POST.get('next') or 'login')
            # return Response(status=status.HTTP_400_BAD_REQUEST)
            except (jwt.exceptions.InvalidKeyError):
                return Response({'error': 'invalid key'}, status = status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({'error': str(e)}, status = status.HTTP_400_BAD_REQUEST)


@api_view(('POST',))
def login(request):
    if request.method == 'POST':

        try:
            subsr = request.data['subsr'] # input 가져오기
            # id = request.POST.get('id')
            # user = Userinfo.objects.filter(subsr = subsr).first() #db에서 데이터 가져오기
            
            #먼저 user에 있는지 없는지 확인
            user = Userinfo.objects.filter(subsr = subsr).first()
            new_user = 0
            if user is None: #예외처리1 - 일치하는 데이터가 없는 경우
                 return Response(
                    {'message': "회원 가입 후 이용해 주세요."},
                    status = status.HTTP_400_BAD_REQUEST
                )
            else:
                #vod 시청 이력은 없고 cont 로그만 있는 경우 
                if VODLog.objects.filter(subsr = subsr).first() is None:
                    new_user = 1
                # user = Userinfo.objects.filter(subsr = subsr).first() #db에서 데이터 가져오기


            serializer = UserSerializer(user)
            user.id = user.subsr_id
            token = TokenObtainPairSerializer.get_token(user) #refresh token 생성
            refresh_token = str(token) #token 문자열화
            access_token = str(token.access_token)
            #활성화 toggle
            user.is_active = True
            user.save()
            #ip 정보 get
            x_fowarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_fowarded_for:
                ip = x_fowarded_for.split(',')[0]
            else:
                ip = request.META.get('REMOTE_ADDR')
            
          
        except KeyError: # json 형식이 잘못 넘어온 경우
            return Response(
                {"message":"아이디를 입력해 주세요"},
                status= status.HTTP_400_BAD_REQUEST
            )

        except Exception as e: #예외 처리
             if "'NoneType'" in str(e) or 'expected a number' in str(e): #숫자 형식 이외의 입력이 있는 경우
                 return Response(
                     {"message":"아이디는 숫자 형태 입니다."},
                     status= status.HTTP_400_BAD_REQUEST
                 )
             if "Expecting value" in str(e): #id 입력이 없는 경우
                 return Response(
                     {"message": "아이디는 필수 입니다."},
                     status = status.HTTP_400_BAD_REQUEST
                 )
             else: #이외 예외 상황
                return Response(
                    {'message':'아이디 확인 후 다시 입력해 주세요.',
                     "error": str(e)},
                    status= status.HTTP_400_BAD_REQUEST
                )
        else: # 예외 없는 경우 - 쿠키에 토큰 저장
            if new_user == 1:
                res = Response(
                    {
                        "user": serializer.data,
                        "ip": ip,
                        "message": 1,
                        "token": {
                            "access": access_token,
                            "refresh": refresh_token,
                        },
                    },
                    status= status.HTTP_200_OK
                )
            else:
                res = Response(
                    {
                        "user": serializer.data,
                        "ip": ip,
                        "message": 0,
                        "token": {
                            "access": access_token,
                            "refresh": refresh_token,
                        },
                    },
                    status = status.HTTP_200_OK,
                )

            #front 연결
            # res = HttpResponse(render(request, 'login.html', {'id': id}))
            res.set_cookie("access", access_token, httponly=True)
            res.set_cookie("refresh", refresh_token, httponly= True)

            return res
            

class SignupAPIView(APIView):
    def post(self, request):

        # x_fowarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

        # if x_fowarded_for:
        #     ip = x_fowarded_for.split(',')[0]
        # else:
        #     ip = request.META.get('REMOTE_ADDR')
            

        ip = socket.gethostbyname(socket.gethostname())

        request.data['ip'] = ip
        serializer = SingupSerializer(data = request.data)
        if serializer.is_valid():
            user = serializer.save()
            #jwt token 접근
            token = TokenObtainPairSerializer.get_token(user)
            refresh_token = str(token)
            access_token = str(token.access_token)
            res = Response(
                {
                    "user": serializer.data,
                    "ip": ip,
                    "message":"register success",
                    "token":{
                        "access": access_token, 
                        "refresh": refresh_token,
                    },
                },
                status= status.HTTP_200_OK,
            )
            #cookie에 넣어주기
            res.set_cookie("access", access_token, httponly= True)
            res.set_cookie("refresh", refresh_token, httponly= True)
            return res
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    #user 정보 확인
    def get(self, request):
        try:
            #access token decode -> id 추출 = user 식별
            access = request.COOKIES['access']
            payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
            pk = payload.get('user_id')
            user = get_object_or_404(Userinfo, pk = pk)
            serializer = UserSerializer(instance = user)
            return Response(serializer.data, status= status.HTTP_200_OK)
        except(jwt.exceptions.ExpiredSignatureError):
            #token 만료 시 갱신
            data = {'refresh': request.COOKIES.get('refresh', None)}
            serializer = TokenObtainPairSerializer(data = data)
            if serializer.is_valid(raise_exception=True):
                access = serializer.data.get('access', None)
                refresh = serializer.data.get('refresh', None)
                payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
                pk = payload.get('user_id')
                serializer = UserSerializer(instance = user)
                res = Response(serializer.data, status=status.HTTP_200_OK)
                res.set_cookie('access', access)
                res.set_cookie('refresh', refresh)
                return res
            return jwt.exceptions.InvalidTokenError
        except(jwt.exceptions.InvalidTokenError):
            #사용 불가 토큰인 경우
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
    #로그인
    def post(self, request):
        id = request.data['id']
        user = Userinfo.objects.filter(id = id).first()

        # x_fowarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

        # if x_fowarded_for:
        #     ip = x_fowarded_for.split(',')[0]
        # else:
        #     ip = request.META.get('REMOTE_ADDR')

        ip = socket.gethostbyname(socket.gethostname())


        #user 존재 X
        if user is None:
            return Response(
                {'message': "id Not exists."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # if user.ip != ip:
        #     return Response(
        #         {"message": "if you changed ur set top box please make a call to CS service center"}, 
        #         status=status.HTTP_303_SEE_OTHER
        #     )

        if user is not None:
            serializer = UserSerializer(user)
            token = TokenObtainPairSerializer.get_token(user) #refresh token 생성
            refresh_token = str(token) #token 문자열화
            access_token = str(token.access_token)

            user.is_active = True
            user.save()

            x_fowarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

            if x_fowarded_for:
                ip = x_fowarded_for.split(',')[0]
            else:
                ip = request.META.get('REMOTE_ADDR')


            res = Response(
                {
                    "user": serializer.data,
                    "ip": ip,
                    "message": "login success",
                    "token": {
                        "access": access_token,
                        "refresh": refresh_token,
                    },
                },
                status = status.HTTP_200_OK,
            )

            res.set_cookie("access", access_token, httponly=True)
            res.set_cookie("refresh", refresh_token, httponly= True)
            return res
        else:
            return Response(
                {"message": "login failed"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
    def delete(self, request):
        update_user = Userinfo.objects.get(id = request.data['id'])
        update_user.is_active = False
        update_user.save()

        #cookie에 저장된 token 삭제 -> logout 처리
        res = Response({
            "message": "Log out success"
        }, status= status.HTTP_202_ACCEPTED)
        res.delete_cookie('refresh')
        return res
    

from .model_code import GenreBasedRecommendationModel, trend_vod, LightFM_Model, watch_series
import pickle
import os
from datetime import datetime, timedelta
import pandas as pd


class VodrecView(APIView):
    def get(self, request):
        # access = request.COOKIES['access']
        access = request.headers.get('Authorization', None)
        genres = request.headers.get('Data', None)

        if access is None:
            return Response({"error": 'access is none'}, status= status.HTTP_400_BAD_REQUEST)
        
        try:
            payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
            pk = payload.get('user_id')
            user = get_object_or_404(Userinfo, pk = pk)
            serializer = UserSerializer(instance = user)
            subsr = serializer.data.get('subsr', None)
            user_id = serializer.data.get('subsr_id', None)
            if user_id == None:
                return Response({"error": "can't find user id"}, status = status.HTTP_400_BAD_REQUEST)
        except (jwt.exceptions.ExpiredSignatureError):
            return Response({"error": "scheck - get new access token plz"}, status= status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status = status.HTTP_400_BAD_REQUEST)
        # kids와 함께 시청하는지 여부
        kids = serializer.data.get('kids', None)
        only_kids = False

        #vod 로그 조회 했을 때 시청한 프로그램이 키즈 밖에 없다면 시간대 별로 다르게 추천 결과 띄우는 것이 의미 없음 -> 변수 생성
        kids_check_queryset = VODLog.objects.filter(subsr_id = user_id).values('age_limit', 'ct_cl')
        kids_check_al = [kids_check_queryset[i].get('age_limit') for i in range(len(kids_check_queryset))]
        kids_check_ct = [kids_check_queryset[i].get('ct_cl') for i in range(len(kids_check_queryset))]
        if set(kids_check_al) == {'키즈'} or set(kids_check_ct) == {'키즈'}:
            only_kids = True

        dec_today  = datetime.now()
        now_time = dec_today.time()
        # now_time = datetime(2023,9,30,21,0,0).time()
        kidstime_start1 = datetime.strptime("8:00:00", "%H:%M:%S").time()
        kidstime_end1 = datetime.strptime("10:00:00", "%H:%M:%S").time()
        kidstime_start2 = datetime.strptime("18:00:00", "%H:%M:%S").time()
        kidstime_end2 = datetime.strptime("20:00:00", "%H:%M:%S").time()

        if os.path.isdir('ml_models'):
            return Response({"error": "no folder"}, status = status.HTTP_400_BAD_REQUEST)

        #1: 장르 유사도 모델 - 기존 vs 신규
        if os.path.isfile('service/ml_models/model_genre.pkl'):
            with open('service/ml_models/model_genre.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            return Response({"message": "can't find model "}, status = status.HTTP_400_BAD_REQUEST)

         #신규 인지 아닌지 -> genres 가 None 인지 아닌지 확인하기 
        if genres is None:
            #아이가 시청하는지 아닌지
            if kids and not only_kids:
                #아이가 시청하는 시간일 경우
                if ((now_time > kidstime_start1) and (now_time < kidstime_end1)) or ((now_time > kidstime_start2) and (now_time < kidstime_end2)):
                    program_ids = model.recommend(user_id, model.score_matrix, 200).program_id
                    li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 1) for i in program_ids]
                    li1 = list(filter(None, li1))[:100]
                else: #아이가 시청하지 않는 시간일 경우
                    program_ids = model.recommend(user_id, model.score_matrix, kids = 1, N = 200).program_id
                    li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]
                    li1 = list(filter(None, li1))[:100]
                    # vlog = VODLog.objects.all().annotate(
                    #     fuse_tms = Cast('use_tms', FloatField()),
                    #     fdisp_rtm_sec = Cast('disp_rtm_sec', FloatField()),
                    #     use_tms_ratio = F('fuse_tms') / F('fdisp_rtm_sec')
                    # ).values('program_id', 'subsr_id', 'use_tms_ratio')
                    # vlog_df = pd.DataFrame(list(vlog), columns = ['program_id', 'subsr_id', 'use_tms_ratio'])
                    # vlog_df.columns = ['program_id', 'subsr_id', 'rating']
                    # vlog_df = vlog_df.sort_values('rating').drop_duplicates(subset = ['program_id'], keep = 'first')
                    # # print(vlog_df.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio'))
                    # score_matrix = model.create_score_matrix(vlog_df)
                    # leng = len(score_matrix) if len(score_matrix) < 250 else 250
                    # program_ids = model.recommend(user_id, score_matrix, leng).program_id
                    # # print(program_ids)
                    # li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 2) for i in program_ids]
                    # li1 = list(filter(None, li1))[:100]
            else: #키즈 데이터 시청 X 경우 - 기본
                program_ids = model.recommend(user_id, model.score_matrix, 100).program_id
                li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]
        else:
            GENRE = ['드라마','액션','모험','코미디','로맨스','애니메이션','스릴러','해외시리즈','멜로','판타지','공포','범죄','SF','미스터리','가족','예능','시대극','다큐','시사교양','키즈']
            # genres = genres[1:-1].split(',')
            if ']' in genres:
                genres = genres[1:-1].split(',')
            else:
                genres = genres.split(',')
            selected_genre = [GENRE[int(i)] for i in genres]
            #아이와 함께 시청하는지 아닌지
            if '키즈' in selected_genre or kids:
                if ((now_time > kidstime_start1) and (now_time < kidstime_end1)) or ((now_time > kidstime_start2) and (now_time < kidstime_end2)):
                    program_ids = model.new_rec(selected_genre, 200).program_id
                    li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 1) for i in program_ids]
                    li1 = list(filter(None, li1))[:100]
                else:
                    program_ids = model.recommend(user_id, model.score_matrix, 250).program_id
                    li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 2) for i in program_ids]
                    li1 = list(filter(None, li1))[:100]
            else:
                program_ids = model.new_rec(selected_genre,100).program_id
                li1 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]

        #2: 랭킹 모델
        vlog = VODSumut.objects.filter(e_bool = 0).values_list('subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month')
        vlog_df = pd.DataFrame(list(vlog), columns = ['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month'])

        today = datetime(2023, 9, 30, 0, 0, 0)
        #mdate = datetime(now())

        program_ids = trend_vod(vlog_df, today, 20).program_id
        li2 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]
        # print(li2)


        #3: 개인화 
        if os.path.isfile('service/ml_models/model_lightfm.pkl'):
            with open('service/ml_models/model_lightfm.pkl', 'rb') as f:
                model2 = pickle.load(f)
        else:
            return Response({"message": "can't find model in filepath"}, status = status.HTTP_400_BAD_REQUEST)

        # lightfm.recommend(subsr, lightfm.score_matrix, lightfm.model, N)

        if kids and ((now_time > kidstime_start1) and (now_time < kidstime_end1)) or ((now_time > kidstime_start2) and (now_time < kidstime_end2)):
            program_ids = model2.recommend(user_id, model2.score_matrix, model2.model, 200).program_id
            li3 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 1) for i in program_ids]
            li3 = list(filter(None, li3))[:100]
        else:
            program_ids = model2.recommend(user_id, model2.score_matrix, model2.model, 100).program_id
        # j = VOD_INFO.objects.filter(program_id__in = program_ids)
            li3 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]


        #4: 최근 본 프로그램과 유사한 프로그램
        if genres is not None:
            li4 = []
        else:
            if kids and not only_kids:
                p_nokids = False
                i = 0
                if ((now_time > kidstime_start1) and (now_time < kidstime_end1)) or ((now_time > kidstime_start2) and (now_time < kidstime_end2)):
                    while p_nokids == False:
                        pname, pid, nokids = VODSumut.objects.filter(subsr_id = user_id).order_by('-log_dt').values_list('program_name', 'program_id', 'nokids')[i]
                        if nokids == 1:
                            p_nokids = False
                        else:
                            p_nokids = True
                        i += 1
                    #장르 유사도 계산
                    genre_vector, genre_similarity = model.calculate_genre_similarity(model.vod_info)
                    #해당 프로그램 세부 장르 데이터 기준 유사도가 높은 프로그램 순으로 데이터 프레임 변경
                    gen_pid = genre_similarity.loc[pid].sort_values(ascending = False).index.to_list()
                    gen_pid.remove(pid)
                    #top n개 정보 추출 후 list 형식 변환
                    #1 -> 성인 프로그램 필터링
                    li4 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i), 1) for i in gen_pid[:100]]
                else: #키즈 시간이 아닌 경우
                    #키즈 프로그램이 아닌 가장 최근 시청한 데이터 추출
                    while p_nokids == False:
                        pname, pid, pgen, page_limit = VODSumut.objects.filter(subsr_id = user_id).order_by('-log_dt').values_list('program_name', 'program_id', 'program_genre', 'age_limit')[i]
                        if '키즈' in pgen or '키즈' in page_limit:
                            p_nokids = False
                        else:
                            p_nokids = True
                        i += 1
                    #장르 유사도 계산
                    genre_vector, genre_similarity = model.calculate_genre_similarity(model.vod_info)
                    #해당 프로그램 세부 장르 데이터 기준 유사도가 높은 프로그램 순으로 데이터 프레임 변경
                    gen_pid = genre_similarity.loc[pid].sort_values(ascending = False).index.to_list()
                    gen_pid.remove(pid)
                    #top n개 정보 추출 후 list 형식 변환
                    li4 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in gen_pid[:100]]
            else:
                pname, pid = VODSumut.objects.filter(subsr_id = user_id).order_by('-log_dt').values_list('program_name', 'program_id')[0]
                #장르 유사도 계산
                genre_vector, genre_similarity = model.calculate_genre_similarity(model.vod_info)
                #해당 프로그램 세부 장르 데이터 기준 유사도가 높은 프로그램 순으로 데이터 프레임 변경
                gen_pid = genre_similarity.loc[pid].sort_values(ascending = False).index.to_list()
                gen_pid.remove(pid)
                #top n개 정보 추출 후 list 형식 변환
                li4 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in gen_pid[:100]]
            

        #5: 현재 시청 중인 프로그램 - 2주 전, 러닝 타임 대비 시청 시간이 80퍼 이하인 경우
        watching_date =  today - timedelta(weeks = 2)

        watching_pids = VODSumut.objects.filter(
            Q(log_dt__range= (watching_date, today)) &
            Q(subsr_id = user_id) 
            ).annotate(
                fuse_tms = Cast('use_tms', FloatField()),
                fdisp_rtm_sec = Cast('disp_rtm_sec', FloatField()),
                use_tms_ratio = F('fuse_tms') / F('fdisp_rtm_sec')
            ).filter(use_tms_ratio__lte = 0.8).values('program_id').distinct()
        
        if watching_pids is None:
            li5 = []
        else:
            li5 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in watching_pids]

        if genres is not None:
            return Response(
                {"data": [li1, li2, li3, li4, li5]},
                status = status.HTTP_200_OK
            )
        return Response(
            {"data": [li1, li2, li3, li4, li5], "current": pname},
            status = status.HTTP_200_OK)


def Voddetail2List(data):
    if data == None:
        return None
    
    li = [data.program_name, data.ct_cl, data.poster_url, data.release_date, data.program_genre, data.age_limit, data.program_id, data.SMRY, data.ACTR_DISP]

    return li

def Vodinfo2Json(data):
    if data == None:
        return None

    dict = {}
    # dict['program_name'] = data.program_name
    # dict['program_id'] = data.program_id
    # dict['program_SMRY'] = data.SMRY

    return dict

def Vodinfo2List(data, kidscheck = 0):
    if data == None:
        return None
    
    if kidscheck == 1:
        if data.nokids:
            return None
        else:
            li = [data.program_id, data.image_id, data.poster_url]
    elif kidscheck == 2:
        if '키즈' in data.program_genre or '키즈' in data.ct_cl or '키즈' in data.age_limit:
            # print('키즈 프로그램')
            return None
        else:
            li = [data.program_id, data.image_id, data.poster_url]
    else:
        li = [data.program_id, data.image_id, data.poster_url]
    return li

#프로그램 상세보기 화면
class VoddetailView(APIView):
    def get(self, request, vod_id):
        video = VOD_INFO.objects.get(program_id = vod_id)
        if video is None:
            return Response({"error": "there's no corresponding program"}, status= status.HTTP_400_BAD_REQUEST)
        li = Voddetail2List(video)

        #장르 기반 모델 읽어오기
        if os.path.isfile('service/ml_models/model_genre.pkl'):
            with open('service/ml_models/model_genre.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            return Response({"message": "can't find model "}, status = status.HTTP_400_BAD_REQUEST)

        #데이터 읽어오기
        # vods = VOD_INFO.objects.filter(e_bool = 0).values_list('program_id', 'program_genre')
        # vod_df = pd.DataFrame(list(vods), columns= ['program_id', 'program_genre'])
        # genre_vector, genre_similarity = model.calculate_genre_similarity(vod_df)

        #장르 유사도 계산
        genre_vector, genre_similarity = model.calculate_genre_similarity(model.vod_info)
        #해당 프로그램 세부 장르 데이터 기준 유사도가 높은 프로그램 순으로 데이터 프레임 변경
        gen_pid = genre_similarity.loc[vod_id].sort_values(ascending = False).index.to_list()
        gen_pid.remove(vod_id)
        #top n개 정보 추출 후 list 형식 변환
        li2 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in gen_pid[:10]]

        return Response({"data": li, "recommend": li2}, status = status.HTTP_200_OK)
    
class CtclView(APIView):
    def get(self, request, ct_cl):
        # 접근 가능 여부 확인
        access = request.headers.get('Authorization', None)

        if access is None:
            return Response({"error": 'access is none'}, status= status.HTTP_400_BAD_REQUEST)
        
        try:
            payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
            pk = payload.get('user_id')
            user = get_object_or_404(Userinfo, pk = pk)
            serializer = UserSerializer(instance = user)
            # subsr = serializer.data.get('subsr', None)
            user_id = serializer.data.get('subsr_id', None)
            if user_id == None:
                return Response({"error": "can't find user id"}, status = status.HTTP_400_BAD_REQUEST)
        except (jwt.exceptions.ExpiredSignatureError):
            return Response({"error": "scheck - get new access token plz"}, status= status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status = status.HTTP_400_BAD_REQUEST)

        # selected_ctcl = ct_cl
        #인코딩 문제 발생시 변경 할 코드
        selected_ctcl = '영화' if ct_cl == 'movie' else 'TV드라마' if ct_cl == 'drama' else None

        if selected_ctcl is None:
            return Response({"error": "invalid ct_cl"}, status = status.HTTP_400_BAD_REQUEST)
        
        #전체 df 생성 - 선택한 ct_cl에 해당하는
        vlog = VODSumut.objects.filter(e_bool = 0, ct_cl = selected_ctcl).values_list('subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month')
        #DB에 없는 ct_cl을 받은 경우
        if vlog is None or not vlog.exists():
            return Response({"message": "invalid ct_cl"}, status=status.HTTP_400_BAD_REQUEST)
        vlog_df = pd.DataFrame(list(vlog), columns = ['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month'])

        #랭킹 모델 - 날짜 지정
        mdate = datetime(2023, 9, 30, 0, 0, 0)
        program_ids = trend_vod(vlog_df, mdate, 20).program_id
        li = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]

        #관심 있는 세부 장르
        Drama_popu_pgens = ['해외시리즈','드라마','코미디','로맨스','액션','미스터리','범죄','판타지','시대극','가족','스릴러','SF','멜로','모험','역사','애니메이션','공포',
                            '정치','복수','추리','성장']
        Movie_popu_pgens = ['드라마','액션','모험','코미디','스릴러','로맨스','멜로','공포','판타지','애니메이션','범죄','SF','미스터리','가족','성인','시대극','다큐',
                            '전쟁','무협','성장','단편','음악','스포츠','역사','학교','공연','사회']
        vod_program = VODLog.objects.filter(subsr_id = user_id, ct_cl = selected_ctcl).values('program_id').distinct().values('program_id', 'log_dt')
        cont_program = CONlog.objects.filter(subsr_id = user_id, ct_cl = selected_ctcl).values('program_id').distinct().values('program_id', 'log_dt')
        
        genre_N = 6 if selected_ctcl == 'TV드라마' else 7

        if vod_program.exists() or cont_program.exists():
            combined_querysets = list(chain(vod_program, cont_program))
            sorted_results = sorted(combined_querysets, key=itemgetter('log_dt'), reverse= True)
            distinct_results = [next(group) for _, group in groupby(sorted_results, key=itemgetter('program_id'))]

            # pgens = distinct_results.order_by('-log_dt').values('program_genre')
            pids = [i.get('program_id') for i in distinct_results]
            pgens = VOD_INFO.objects.filter(program_id__in = pids).values('program_genre')
            pgens_df = pd.DataFrame(list(pgens), columns = ['program_genre'])['program_genre'].str.get_dummies(sep = ', ')
            pgens_li = pgens_df.sum().sort_values(ascending= False).index.to_list()
            popu_pgens = Drama_popu_pgens if selected_ctcl == 'TV드라마' else Movie_popu_pgens
            pgens_li = [i for i in pgens_li if i in popu_pgens] #개수 확인
            if len(pgens_li) < 10: #unique 프로그램 기준 세부 자을 보기를 10개 이하로 한 유저에 대해서는 ct_cl이 드라마 & 세부 장르가 가장 많은 순서의 list 추가
                popu_pgens = [i for i in popu_pgens if i not in pgens_li]
                pgens_li = pgens_li + popu_pgens
            pgens_li = pgens_li[:genre_N]
        else:
            pgens_li = Drama_popu_pgens[:genre_N] if selected_ctcl == 'TV드라마' else Movie_popu_pgens[:genre_N]

        #몰아보기 - 드라마
        if selected_ctcl == 'TV드라마':
            #vod sumut에서 가져오는 경우
            # ru = VODLog.objects.filter(
            #     ct_cl = selected_ctcl
            # ).values(
            #     'month', 'day', 'subsr_id', 'program_id', 'episode_num'
            # ).annotate(
            #     sum_ut = Sum('use_tms'),
            #     remain_ut = F('sum_ut') / F('disp_rtm_sec')
            # ).filter(
            #     remain_ut__gt = 1
            # ).values('remain_ut') 
        
            # addvs = VODLog.objects.filter(
            #     ct_cl = selected_ctcl
            # ).values(
            #     'month', 'day', 'subsr_id', 'program_id', 'episode_num'
            # ).annotate(
            #     sum_ut = Sum('use_tms'),
            #     remain_ut = F('sum_ut') / F('disp_rtm_sec')
            # ).filter(
            #     remain_ut__gt = 1
            # ).values('subsr_id', 'program_id')

            # addvs_df = pd.DataFrame(list(addvs), columns=['subsr_id', 'program_id'])
            # # print(pd.concat([addvs_df, pd.DataFrame([addvs_df.loc[0]] * ru[0].get('remain_ut'))], ignore_index= True))
            # for i in range(len(ru)):
            #     addvs_df = pd.concat([addvs_df, pd.DataFrame([addvs_df.loc[i]] * (ru[i].get('remain_ut') -1))], ignore_index= True)


            # vs = VODSumut.objects.values('subsr_id', 'program_id')
            # vs_df = pd.DataFrame(list(vs), columns= ['subsr_id', 'program_id'])

            # result = pd.concat([vs_df, addvs_df], ignore_index= True)

            # li2 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in watch_series(result, 100)]
            # ------------------------------------
            #vod log에서 가져오는 경우

            #몰아보기
            series_topN = 100 #몰아보기 top N
            result = VODLog.objects.values('program_id').filter(
                ct_cl = selected_ctcl
            ).annotate(
                program_count = Cast(Count('program_id'), FloatField()),
                subsr_count = Cast(Count('subsr_id', distinct= True), FloatField()),
            ).annotate(
                ratio = F('program_count') / F('subsr_count')
            ).order_by(
                '-ratio'
            )[:series_topN].values('program_id')

            li2 = [Vodinfo2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in result]
            

            #드라마 세부 장르 추천
            #유저가 관심있어하는 프로그램 세부 장르 list - 신규도 cont는 있기 때문에 ct_cl 홈 사용 가능
            li3 = []
            for i in pgens_li:
                tmp = random.sample(list(VOD_INFO.objects.filter(ct_cl = selected_ctcl, program_genre__icontains = i).values('program_id')), 10)
                tmp_li = [Vodinfo2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in tmp]
                li3.append(tmp_li)
            # li = [Vodinfo2List(VOD_INFO.objects.get(program_id = i)) for i in program_ids]
            print(len(pgens_li))
            return Response({"data": [li, li2, li3],"genres": pgens_li}, status = status.HTTP_200_OK)

        else:
            li2 = []
            for i in pgens_li:
                tmp = random.sample(list(VOD_INFO.objects.filter(ct_cl = selected_ctcl, program_genre__icontains = i).values('program_id')), 10)
                tmp_li = [Vodinfo2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in tmp]
                li2.append(tmp_li)
            
            return Response({"data": [li, li2], "genres":pgens_li}, status = status.HTTP_200_OK)

# 
class Vodrec_SimpleView(APIView):
    def get(self, request, btn_selected):
        
        #추출한 데이터 수
        N = 10
        if btn_selected == 1:
        
            #음악 / 트로트 - 11개
            query_set = VOD_INFO.objects.filter(program_genre__icontains =('음악'))
            query_set2 = query_set.filter(
                Q(program_name__icontains = ('트롯')) |
                Q(ACTR_DISP__icontains = ('김성주')) |
                Q(ACTR_DISP__icontains = ('임영웅')) |
                Q(ACTR_DISP__icontains = ('장윤정')) |
                Q(ACTR_DISP__icontains = ('송가인')) |
                Q(SMRY__icontains = ('트롯'))
            )#.values('program_id')

            # music_pid = random.sample(list(query_set2.union(query_set).values('program_id')), 10)
            music_pid = query_set2.union(query_set).values('program_id')[:N]
            # print(len(music_pid))
            li1 = [Voddetail2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in music_pid]

            return Response({"data": li1}, status= status.HTTP_200_OK)
        elif btn_selected == 2:
            #건강 / 음식 - 72개
            diet = random.sample(list(VOD_INFO.objects.filter(
                (~Q(ct_cl = 'TV애니메이션') |
                ~Q(program_genre__icontains = ('애니메이션'))) &
                Q(program_genre__icontains = ('건강')) |
                Q(program_genre__icontains = ('요리')) |
                # Q(SMRY__icontains = ('요리')) |
                # Q(SMRY__icontains = ('건강')) |
                Q(ACTR_DISP__icontains = ('백종원'))
            ).values('program_id')), N)

        # print(len(VOD_INFO.objects.filter(
        #     Q(program_genre__icontains = ('건강')) |
        #     Q(program_genre__icontains = ('요리')) |
        #     Q(SMRY__icontains = ('요리')) |
        #     Q(SMRY__icontains = ('건강')) |
        #     Q(ACTR_DISP__icontains = ('백종원'))
        # ).values('program_id')))

            li2 = [Voddetail2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in diet]
            return Response({"data": li2}, status = status.HTTP_200_OK)
        elif btn_selected == 3:
            #드라마 - 몰아보기
            # series_topN = 100 #몰아보기 top N
            series_drama = random.sample(list(VODLog.objects.values('program_id').filter(
                ct_cl = 'TV드라마'
            ).annotate(
                program_count = Cast(Count('program_id'), FloatField()),
                subsr_count = Cast(Count('subsr_id', distinct= True), FloatField()),
                ratio = F('program_count') / F('subsr_count')  
            ).order_by(
                '-ratio'
            ).values('program_id')), N)

            li3 = [Voddetail2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in series_drama]

            return Response({"data": li3}, status = status.HTTP_200_OK)
        else:
            return Response({"message": "invalid choice"}, status =status.HTTP_400_BAD_REQUEST)
        return Response(status = status.HTTP_200_OK)
        # return Response({"data" : [li1, li2, li3]}, status= status.HTTP_200_OK)

from django.db.models import Value
from django.db.models.functions import Replace


def custom_sort_key(word, priority_words):
    pname = word['program_name']
    return (pname not in priority_words, pname)

class SearchVODView(APIView):
    def get(self, request):

        search_item = request.GET.get('Searchword', None)

        if search_item is None:
            return Response(status=status.HTTP_204_NO_CONTENT)

        search_item = search_item.replace(' ', '')

        # print(search_item)
        pids = VOD_INFO.objects.all().annotate(
                string_field=Replace('program_name', Value(' '), Value(''))
                # programs = 'program_name'.replace(" ", "")
            ).filter(
            Q(string_field__contains = (search_item)) |
            Q(string_field__icontains = (search_item)) |
            Q(ACTR_DISP__icontains = (search_item)) #|
            # Q(program_name__iregex=r'\y{}\y'.format(search_item))
        ).values('program_name', 'program_id').distinct()

        


        # Sort the list of dictionaries using the custom sorting key
        # sorted_list_of_dicts = sorted(pids, key=custom_sort_key(search_item))
        sorted_list_of_dicts = sorted(pids, key=lambda x: custom_sort_key(x, search_item))


        if len(sorted_list_of_dicts) > 50:
            sorted_list_of_dicts = sorted_list_of_dicts[:50]
        li = [Vodinfo2List(VOD_INFO.objects.get(program_id = i.get('program_id'))) for i in pids]
        #제목, 배우 icontains + smry icontains
        #띄어 쓰기 - 검색 사항에서 제외 하는 법
        #program_id + poster_id 보내기 
        return Response({"data": li}, status= status.HTTP_200_OK)
    
import json
import boto3
from botocore.exceptions import NoCredentialsError

class ChartView(APIView):
    def get(self, request):

        #버킷 정보 설정
        # aws_access_key_id = 
        # aws_secret_access_key = 
        s3_bucket_name = 'airflowexample'

        #client 생성
        s3_client = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)

        #Json 파일 다운로드
        try:
            response = s3_client.get_object(Bucket = s3_bucket_name, Key = 'recommendation.json')
            json_data = json.loads(response['Body'].read().decode('utf-8'))
        except NoCredentialsError:
            #aws 인정 정보가 유효하지 않을 때 - 예외 처리
            return Response({"error": "invalid credentials"}, status= status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status= status.HTTP_400_BAD_REQUEST)
        
        if json_data is None:
            return Response({"error": "got no data"}, status= status.HTTP_400_BAD_REQUEST)
        #JSON 데이터 -> HTML 전달
        print(json_data)

        # 나중 로직 
        #1 s3데이터를 가져와서 db에 저장하는 function 따로 하나 만들기 
        #2 위에 function 호출
        #3 DB table에서 정해진 만큼(log 기준) 데이터 가져오기
        #4 형식 변환해서 return 하기
        
        # for i,j in json_data.items():
        #     print(i.values(),j )
        
        # print(len(json_data.get('columns')))
        performance_dict = [Performance2Json(json_data.get('columns')[i], '0930', json_data.get('values')[i]) for i in range(len(json_data.get('columns')))]

        # print(performance_dict)
        return Response({"data": performance_dict}, status= status.HTTP_200_OK)

def get_cr_dict(data):

    result_dict = {}
    for entry in data:
        program_id = entry['program_id']
        subsr_id = entry['subsr_id']

        result_dict.setdefault(program_id, []).append(subsr_id)
        # if program_id not in result_dict:
        #     result_dict[program_id] = [subsr_id]
        # else:
        #     result_dict[program_id].append(subsr_id)

    return result_dict

# def get_conversionrate(date_info):
#     cont_cr = list(CONlog.objects.filter(
#         log_dt__range = (date_info[0], date_info[1])
#     ).values('program_id', 'subsr_id').distinct())
#     vod_cr = list(VODLog.objects.filter(
#         log_dt__range = (date_info[0], date_info[1])
#     ).values('program_id', 'subsr_id').distinct())

#     # print(cont_cr)

#     cont_cr_dict = get_cr_dict(cont_cr)
#     vod_cr_dict = get_cr_dict(vod_cr)

    

#     c_pid = [i.get('program_id')for i in cont_cr]
#     v_pid = [i.get('program_id') for i in vod_cr]

#     # print(vod_cr_dict)

#     pids = set(c_pid).union(set(v_pid))
    
#     cont_cr_dict2 = {'beginner':{pid: 0 for pid in pids},
#                      'starter': {pid: 0 for pid in pids},
#                      'standard': {pid:0 for pid in pids},
#                      'heavy': {pid:0 for pid in pids}}
    
#     vod_cr_dict2 = {'beginner':{pid: 0 for pid in pids},
#                      'starter': {pid: 0 for pid in pids},
#                      'standard': {pid:0 for pid in pids},
#                      'heavy': {pid:0 for pid in pids}}

    # print(vod_cr_dict2['beginner'][3366])
    # for i in pids:
    #     vod_subsrs = vod_cr_dict.get(i, 0)
    #     if vod_subsrs == 0:
    #         clusters = list(Userinfo.objects.filter(subsr_id__in = vod_subsrs).values('cluster'))
    #         clusters = [i.get('cluster') for i in clusters]
    #         for i in clusters:
    #             if i == 1:
    #                 vod_cr_dict2['beginner'][i] += 1
    #             elif i == 2:
    #                 vod_cr_dict2['starter'][i] += 1
    #             elif i == 3:
    #                 vod_cr_dict2['standard'][i] += 1
    #             elif i == 4:
    #                 vod_cr_dict2['heavy'][i] +=1

    # return None
from datetime import date

class ChartsampleView(APIView):
    def get(self, request):

        ##1번
        #오늘 방문자 수 - vod log, cont log subsr unique 합
        #active 사용자 수 - vod log subsr nunique
        #오늘 전환율 
        today = datetime(2023, 9, 24).date()
        start_time = datetime(2023, 9, 24,0,0,0).time()
        end_time = datetime.now().time()
        today_start = datetime.combine(today, start_time)
        today_end = datetime.combine(today, end_time)

        viewer = len(CONlog.objects.filter(log_dt__range = (today_start, today_end)).values('subsr_id').distinct())
        active_user = len(VODLog.objects.filter(log_dt__range = (today_start, today_end)).values('subsr_id').distinct())

        # today_conversionrate = 
        cont_cr = list(CONlog.objects.filter(
            log_dt__range = (today_start, today_end)
            ).values('program_id').distinct().annotate(subsr_count = Count('subsr_id')))
        vod_cr = list(VODLog.objects.filter(
            log_dt__range = (today_start, today_end)
        ).values('program_id').distinct().annotate(subsr_count = Count('subsr_id')))
        # print(list(cont_cr))

        c_pid = [i.get('program_id')for i in cont_cr]
        v_pid = [i.get('program_id') for i in vod_cr]

        pids = set(c_pid).union(set(v_pid))

        cont_cr_dict = {entry['program_id']: entry['subsr_count'] for entry in cont_cr}
        vod_cr_dict = {entry['program_id']: entry['subsr_count'] for entry in vod_cr}

        # print(cont_cr_dict)
        today_conversionrate_dict = {}
        for i in pids:
            vod_subsr_count = vod_cr_dict.get(i, 0)
            cont_subsr_count = cont_cr_dict.get(i, 1)

            today_conversionrate_dict[i] = round(vod_subsr_count / cont_subsr_count,2)


        max_key = max(today_conversionrate_dict, key=lambda k: today_conversionrate_dict[k])
        max_program = VOD_INFO.objects.filter(program_id = max_key).values('program_name')

        today_conversionrate = sum(today_conversionrate_dict.values()) / len(today_conversionrate_dict)


        ##2번
        # user cluster 별로 전환율 계산 -> 막대 그래프
        #user cluster 1-4까지 (beginner, starter, standard, heavy)
        #cont log / 
        #전환율은 daily로 하면 높지 않기 때문에 주차로 그려서 한 달 그래프 그리기

        dates = []
        for i in list(range(0,4))[::-1]:
            start_date = today - timedelta(days=today.weekday()) - timedelta(weeks=i)
            end_date = start_date + timedelta(days = 6)
            dates.append([start_date, end_date])
        
        # print(dates)
        cont_cr = CONlog.objects.filter(
            log_dt__range = (dates[0][0], dates[-1][1])
        ).values('program_id', 'subsr_id').distinct().values('log_dt', 'program_id', 'subsr_id')

        vod_cr = VODLog.objects.filter(
            log_dt__range = (dates[0][0], dates[-1][1])
        ).values('program_id','subsr_id').distinct().values('log_dt','program_id', 'subsr_id')

        users = Userinfo.objects.values('subsr_id', 'cluster')

        cont_cr_df = pd.DataFrame(list(cont_cr), columns = ['log_dt', 'program_id', 'subsr_id'])
        vod_cr_df = pd.DataFrame(list(vod_cr), columns = ['log_dt', 'program_id', 'subsr_id'])
        users_df = pd.DataFrame(list(users), columns = ['subsr_id', 'cluster'])

        cont_cr_df = cont_cr_df.merge(users_df, how = 'left', on = 'subsr_id')
        vod_cr_df = vod_cr_df.merge(users_df, how = 'left', on = 'subsr_id')

        cont_cr_df['log_dt'] = pd.to_datetime(cont_cr_df['log_dt'])
        vod_cr_df['log_dt'] = pd.to_datetime(vod_cr_df['log_dt'])

        pids = set(cont_cr_df.program_id.to_list()).union(set(vod_cr_df.program_id.to_list()))

        whole_crs = []
        for i in dates:
            cdf = cont_cr_df.loc[(cont_cr_df['log_dt'].dt.date >= i[0]) & (cont_cr_df['log_dt'].dt.date <= i[1])]
            vdf = vod_cr_df.loc[(vod_cr_df['log_dt'].dt.date >= i[0]) & (vod_cr_df['log_dt'].dt.date <= i[1])]
            # print(cdf)
            cluster_crs = []
            for c in range(0,4):
                cdf_ = cdf[cdf.cluster == c].groupby('program_id').count().subsr_id.rename('subsr_count').reset_index()
                cdict = dict(zip(cdf_['program_id'].astype(str), cdf_['subsr_count']))
                # print(cdict)
                vdf_ = vdf[vdf.cluster == c].groupby('program_id').count().subsr_id.rename('subsr_count').reset_index()
                vdict = dict(zip(vdf_['program_id'].astype(str), vdf_['subsr_count']))
                cr_dict = {}
                for i in pids:
                    vod_subsr_count = vdict.get(str(i),0)
                    cont_subsr_count = cdict.get(str(i), 1)
                    cr_dict[i] = round(vod_subsr_count / cont_subsr_count,2)
                cluster_crs.append(sum(cr_dict.values()) / len(cr_dict))
            whole_crs.append(cluster_crs)

        monthly_conversionrate = [{
            "date": ' - '.join([d.strftime('%Y.%m.%d') for d in i]),
            "beginner": j[0],
            "starter": j[1],
            "standard": j[2],
            "heavy": j[3] 
        }for i,j in zip(dates, whole_crs)]

        ##3번
        #model performance -> line 그래프
        #S3에서 값 가져와서 그래프 그리기
        #S3에서 날짜 데이터도 가져와서 string으로 x 값에 넣어주기

        ##4번
        #ct_cl 인기도 -> pie chart
        #인기도는 랭킹이 주차 별인 것을 감안 + 지금 hot trending을 위해서 
        #daily, weekly로 그리기

        ##5번
        #genre 인기도 -> pie chart
        #똑같은 이유로 daily, weekly로 그리기

        
        mdate = datetime(2023, 9, 30, 23, 23, 59)
        daily = mdate - timedelta(days = 1)

        vod_subsrs = VODLog.objects.filter(
            log_dt__range = (daily, mdate)
        ).values(
            'subsr_id'
        ).distinct()
        cont_subsrs = CONlog.objects.filter(
            log_dt__range = (daily, mdate)
        ).values('subsr_id').distinct()

        total_subsrs = list(chain(vod_subsrs, cont_subsrs))
        total_subsrs = [next(group) for _, group in groupby(total_subsrs, key=itemgetter('subsr_id'))]


        items = VODLog.objects.filter(
            log_dt__range = (daily, mdate)
        ).annotate(
            fuse_tms = Cast('use_tms', FloatField()),
            fdisp_rtm_sec = Cast('disp_rtm_sec', FloatField()),
            use_tms_ratio = F('fuse_tms') / F('fdisp_rtm_sec')
        ).values(
            'subsr_id',
            'program_genre',
            'use_tms_ratio',
            'ct_cl'
        )

        genres = ['SF', '가족', '게임', '경쟁','공포','단편','동물','드라마','로맨스','리얼리티','멜로','모험','무협', '미스터리','범죄','법정', '복수', '스릴러', '스포츠','시대극','시사교양','애니메이션', 
                  '액션', '예능', '음악', '의학', '일상', '코미디', '크리에이터', '키즈', '토크', '해외시리즈']
        
        items_df = pd.DataFrame(list(items), columns = ['subsr_id', 'program_genre', 'use_tms_ratio', 'ct_cl'])


        gen_cuser = [items_df[items_df.program_genre.str.contains(i)].subsr_id.nunique() for i in genres]
        # gen_puser = [sum(items_df[items_df.program_genre.str.contains(i)].use_tms_ratio.values) / items_df[items_df.program_genre.str.contains(i)].subsr_id.nunique() for i in genres]
        gen_pt = [sum(items_df[items_df.program_genre.str.contains(i)].use_tms_ratio.values) / len(items_df[items_df.program_genre.str.contains(i)]) for i in genres]


        gen_count = [round(i*j, 2) for i,j in zip(gen_pt, gen_cuser)]

        gen_dict = {i:j for i,j in zip(genres, gen_count)}
        gen_dict = dict(sorted(gen_dict.items(), key=lambda item: item[1], reverse=True))
        gen_dict = {k: gen_dict[k] for k in list(gen_dict.keys())[:10]}
        # gen_dict =sorted(gen_dict, key = lambda x: x[1])
        gen_data = [genre_chart(i, j) for i,j in gen_dict.items()]

        ct_cls = items_df.ct_cl.unique()
        ctcl_cuser = [items_df[items_df.ct_cl == i].subsr_id.nunique() for i in ct_cls]
        ctcl_pt = [sum(items_df[items_df.ct_cl == i].use_tms_ratio.values) / len(items_df[items_df.ct_cl == i]) for i in ct_cls]

        ctcl_count = [round(i*j, 2) for i,j in zip(ctcl_pt, ctcl_cuser)]
        ctcl_dict = {i:j for i,j in zip(ct_cls, ctcl_count)}
        ctcl_dict = dict(sorted(ctcl_dict.items(), key = lambda item: item[1], reverse= True))
        ctcl_data = [genre_chart(i,j) for i,j in ctcl_dict.items()]

        
        #버킷 정보 설정
        # aws_access_key_id = 
        # aws_secret_access_key = 
        s3_bucket_name = 'hello00.net-airflow'
        performance_folder = 'model_accuracy'

        #client 생성
        s3_client = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)

        #Json 파일 다운로드
        try:
            #get datafile
            all_div = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/all_diversity.json')
            map = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/map.json')
            mar = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/mar.json')
            precision = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/precision.json')
            recall = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/recall.json')
            test_diversity = s3_client.get_object(Bucket = s3_bucket_name, Key = f'{performance_folder}/test_diversity.json')

            #decode data 
            all_diversity_data = json.loads(all_div['Body'].read().decode('utf-8'))
            map_data = json.loads(map['Body'].read().decode('utf-8'))
            mar_data = json.loads(mar['Body'].read().decode('utf-8'))
            precision_data = json.loads(precision['Body'].read().decode('utf-8'))
            recall_data = json.loads(recall['Body'].read().decode('utf-8'))
            test_diversity_data = json.loads(test_diversity['Body'].read().decode('utf-8'))

        except NoCredentialsError:
            #aws 인정 정보가 유효하지 않을 때 - 예외 처리
            return Response({"error": "invalid credentials"}, status= status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status= status.HTTP_400_BAD_REQUEST)
        
        if all_diversity_data is None or map_data is None or mar_data is None or precision_data is None or recall_data is None or test_diversity_data is None:
            return Response({"error": "got no data"}, status= status.HTTP_400_BAD_REQUEST)
        #JSON 데이터 -> HTML 전달
        # print(json_data)

        # 나중 로직 
        #1 s3데이터를 가져와서 db에 저장하는 function 따로 하나 만들기 
        #2 위에 function 호출
        #3 DB table에서 정해진 만큼(log 기준) 데이터 가져오기
        #4 형식 변환해서 return 하기
    
        
        # print(len(json_data.get('columns')))
        performance_data = [all_diversity_data, map_data, mar_data, precision_data, recall_data, test_diversity_data]
        
        # print(performance_data[0])
        performance_dict = [Performance2Json(i.get('columns')[0][0], i.get('values')) for i in performance_data]
        # performance_dict = [Performance2Json(performance_data.get('columns')[i], formatted_date, performance_data.get('values')[i]) for i in range(len(performance_data.get('columns')))]



        return Response({"data": [gen_data, performance_dict, ctcl_data, monthly_conversionrate, viewer, active_user, today_conversionrate, max_program]}, status= status.HTTP_200_OK)


#use_tms_ratio / row * user_num => ? 

def genre_chart(data1, data2 = None):
    return {"id": data1, "label": data1, "value": data2}

def Performance2Json(col, y):

#     # Add 7 days to the current datetime
#     new_datetime =
# # Format the new datetime as a string
# formatted_new_datetime =  

    # pattern = re.compile('\(([^)]+)')
    p2 = r'\([^)]*\)'
    # tdate = pattern.findall(str(col))
    # tdate = datetime.strptime(tdate[0], '%Y.%m.%d')
    tdate = datetime(2023,10,1).date()
    # print(tdate)
    # tdate = datetime.strptime(tdate, '%Y-%m-%d')
    # print(tdate)
    dict = {
        "id" : re.sub(p2, repl = '', string = str(col)),
        "data" : [
            {
                "x": (tdate + timedelta(days=7 *idx)).strftime("%Y-%m-%d"),
                "y": vals[0]
            }
            for idx, vals in enumerate(y)
        ]
    }

    return dict
