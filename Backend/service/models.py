from django.db import models

class UserManager():
    def create_user(self, subsr, ip,  **kwargs):
        if not id:
            raise ValueError("ID is required")

        user = self.model(
            subsr = subsr,
            ip = ip,
            **kwargs
        )

        user.save(using = self._db)
        return user
    
    def create_superuser(self, subsr, ip,  **kwargs):
        superuser = self.create_user(
            subsr = subsr,
            ip = ip
        )
        superuser.is_staff = True
        superuser.is_superuser = True
        superuser.is_active = True

        superuser.save(using = self._db)
        return superuser


class Userinfo(models.Model):
    subsr = models.IntegerField(default= None)
    subsr_id = models.IntegerField(primary_key=True) #FK -> USERinfo
    kids = models.BooleanField(default=0) #0이 kids 기록 없는거
    ip = models.CharField(max_length = 25, blank= True, null = True)
    cluster = models.IntegerField(default = None, blank = True, null =True)

    is_superuser = models.BooleanField(default = False)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default = False)

    objects = UserManager()
    USERNAME_FIELD = 'subsr'
    
    class Meta:
        managed = True
        db_table="userinfo"

class VOD_INFO(models.Model):
    id = models.AutoField(primary_key=True, default= 0)
    program_name = models.CharField(max_length=255, default=None, null=True)
    ct_cl = models.CharField(max_length=50, default = None, null=True)
    image_id = models.IntegerField(null=True, default=0)
    poster_url = models.URLField(max_length=1000, default=None, null=True)
    release_date = models.IntegerField(null=True, blank=True)
    program_genre = models.CharField(max_length=255, default=None ,null=True)
    age_limit = models.CharField(max_length=20, default=15, null=True)
    nokids = models.IntegerField(default = 0, null=True)
    program_id = models.CharField(max_length=100, default = None, null=True)
    e_bool = models.BooleanField(default=0, null=True) #0인게 모델 사용(event x)
    SMRY = models.TextField(null =True, default = None)
    ACTR_DISP = models.TextField(null = True, default = None)

    def __str__(self):
        return self.program_name
    class Meta:
        managed = True
        db_table="vodinfo"


# class EVOD_INFO(models.Model):
#     program_name = models.CharField(max_length=255, default=None)
#     ct_cl = models.CharField(max_length=50)
#     poster_id = models.IntegerField(primary_key=True, default=0)
#     poster_url = models.URLField(max_length=1000, default=None)
#     release_date = models.IntegerField(null=True, blank=True)
#     program_genre = models.CharField(max_length=255, default=None)
#     age_limit = models.CharField(max_length=20, default=15)
#     nokids = models.IntegerField(default=0)
#     program_id = models.CharField(max_length=20, default = 000)
#     summary = models.CharField(null = True, max_length=2000)
#     actor = models.CharField(null = True, max_length=100)


#     def __str__(self):
#         return self.program_name
#     class Meta:
#         db_table="evodinfo"


class CONlog(models.Model):
    id = models.AutoField(primary_key=True)
    subsr = models.IntegerField(default = None, null=True)
    SMRY = models.TextField(max_length= 2000, default = None, null=True)
    ACTR_DISP = models.CharField(max_length=10, default = None, null=True)
    disp_rtm = models.CharField(max_length=10, default = None, null=True)
    pinfo = models.CharField(max_length=10, default = None, null=True)
    disp_rtm_sec = models.IntegerField(default = 0, null=True)
    image_id = models.IntegerField(default = 0, null=True)
    episode_num = models.IntegerField(null= True, default = 0)
    log_dt = models.DateTimeField(default = None, null=True)
    year = models.IntegerField(default = 0, null=True)
    month = models.IntegerField(default = 0, null=True)
    day = models.IntegerField(default = 0, null=True)
    hour = models.IntegerField(default= 0, null=True)
    minute = models.IntegerField(default = 0, null=True)
    second = models.IntegerField(default = 0, null=True)
    weekday = models.IntegerField(default = 0, null=True)
    day_name = models.CharField(max_length=20, default = None, null=True)
    subsr_id = models.IntegerField(default = 0, null=True) #FK -> USERinfo
    kids = models.IntegerField(default = 0, null=True)
    program_name = models.CharField(max_length=255, default = None, null=True)
    ct_cl = models.CharField(max_length=50, default = None, null=True)
    release_date = models.IntegerField(null=True, blank=True)
    program_genre = models.CharField(max_length=255, default = None, null=True)
    age_limit = models.CharField(max_length=20, default= None, null=True)
    nokids = models.IntegerField(default=0, null=True)
    program_id = models.IntegerField(default = 0, null=True)
    e_bool = models.IntegerField(default=0, null=True) #0인게 모델 사용(event x)

    class Meta :
        managed = True
        db_table = 'contlog'

class VODLog(models.Model):
    id = models.AutoField(primary_key=True)
    subsr = models.IntegerField(default = 0, null=True)
    use_tms = models.IntegerField(default = 0, null=True)
    SMRY = models.TextField(max_length=2000, default = None, null=True)
    ACTR_DISP = models.CharField(max_length=10, default = None, null=True)
    disp_rtm = models.CharField(max_length=10, default = None,null=True)
    upload_date = models.DateTimeField(default = None,null=True)
    pinfo = models.CharField(max_length=10, default = None,null=True)
    disp_rtm_sec = models.IntegerField(default = 0,null=True)
    image_id = models.IntegerField(default = 0,null=True)
    episode_num = models.IntegerField(default = 0,null=True)
    log_dt = models.DateTimeField(default = None, null=True)
    year = models.IntegerField(default = 0, null=True)
    month = models.IntegerField(default = 0, null=True)
    day = models.IntegerField(default = 0, null=True)
    hour = models.IntegerField(default = 0, null=True)
    minute = models.IntegerField(default = 0, null=True)
    second = models.IntegerField(default = 0, null=True)
    weekday = models.IntegerField(default = 0, null=True)
    day_name = models.CharField(max_length=20, default = None, null=True)
    subsr_id = models.IntegerField(null=True) #FK -> USERinfo
    kids = models.IntegerField(default = 0, null=True)
    program_name = models.CharField(max_length=255, default = None ,null=True)
    ct_cl = models.CharField(max_length=50, default = None)
    release_date = models.IntegerField(null=True, blank=True, default = None)
    program_genre = models.CharField(max_length=255, default = None, null=True)
    age_limit = models.CharField(max_length=20, default = None, null=True)
    nokids = models.IntegerField(default = 0, null=True)
    program_id = models.IntegerField(default = 0, null=True)
    count_watch = models.IntegerField(default = 0, null=True)
    e_bool = models.IntegerField(default=0, null=True) #0인게 모델 사용(event x)

    class Meta:
        managed = True
        db_table = 'vodlog'


class VODSumut(models.Model):
    id = models.AutoField(primary_key=True)
    subsr = models.IntegerField(default = 0,null=True)
    use_tms = models.IntegerField(default = 0,null=True)
    SMRY = models.TextField(max_length=2000, default = None,null=True)
    ACTR_DISP = models.CharField(max_length=10, default = None,null=True)
    disp_rtm = models.CharField(max_length=10, default = None,null=True)
    upload_date = models.DateTimeField(default = None, null=True)
    pinfo = models.CharField(max_length=10, default = None, null=True)
    disp_rtm_sec = models.IntegerField(default = 0)
    image_id = models.IntegerField(default = 0, null=True)
    episode_num = models.IntegerField(default = 0, null=True)
    log_dt = models.DateTimeField(default = None, null=True)
    year = models.IntegerField(default = 0, null=True)
    month = models.IntegerField(default = 0, null=True)
    day = models.IntegerField(default = 0, null=True)
    hour = models.IntegerField(default = 0, null=True)
    minute = models.IntegerField(default = 0, null=True)
    second = models.IntegerField(default = 0, null=True)
    weekday = models.IntegerField(default =0, null=True)
    day_name = models.CharField(max_length=20, default = None, null=True)
    subsr_id = models.IntegerField(null=True, default = 0) #FK -> USERinfo
    # kids = models.IntegerField()
    program_name = models.CharField(max_length=255, null=True, default = None)
    ct_cl = models.CharField(max_length=50, null=True, default = None)
    release_date = models.IntegerField(null=True, blank=True, default = 0)
    program_genre = models.CharField(max_length=255, null=True, default = None)
    age_limit = models.CharField(max_length=20, null=True, default = None)
    nokids = models.IntegerField(default = 0, null=True)
    program_id = models.IntegerField(default = 0, null=True)
    count_watch = models.IntegerField(default = 0, null=True) #vod_sumut에만 적용
    e_bool = models.IntegerField(default=0, null=True) #0인게 모델 사용(event x)

    class Meta:
        managed = True
        db_table = 'vods_sumut'